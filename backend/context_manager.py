import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from collections import deque

class ConversationContext:
    """Manages conversation context for multi-turn dialogue"""
    
    def __init__(self, session_id: str, max_turns: int = 10, memory_type: str = 'full'):
        """
        Parameters:
        - session_id: Unique conversation session ID
        - max_turns: Maximum conversation turns to remember
        - memory_type: 'full' (all), 'summary' (compressed), 'recent' (last N)
        """
        self.session_id = session_id
        self.max_turns = max_turns
        self.memory_type = memory_type
        self.context_buffer = deque(maxlen=max_turns)
        self.context_summary = ""
        self.last_update = datetime.now()
    
    def add_turn(self, user_input: str, bot_response: str) -> None:
        """Add a turn to conversation context"""
        self.context_buffer.append({
            'user': user_input,
            'bot': bot_response,
            'timestamp': datetime.now()
        })
        self.last_update = datetime.now()
    
    def get_context_string(self) -> str:
        """Get formatted context for model input"""
        if self.memory_type == 'full':
            return self._get_full_context()
        elif self.memory_type == 'summary':
            return self._get_summary_context()
        elif self.memory_type == 'recent':
            return self._get_recent_context()
    
    def _get_full_context(self) -> str:
        """Return full conversation history"""
        lines = []
        for turn in self.context_buffer:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Bot: {turn['bot']}")
        
        return "\n".join(lines)
    
    def _get_summary_context(self) -> str:
        """Return compressed context with key topics"""
        if not self.context_buffer:
            return ""
        
        # Extract key entities/topics (simple approach)
        topics = set()
        for turn in self.context_buffer:
            # Extract important words (this is simplified)
            words = turn['user'].split()
            topics.update([w for w in words if len(w) > 5])
        
        summary = f"Topics discussed: {', '.join(list(topics)[:5])}\n"
        
        # Include last turn
        if self.context_buffer:
            last = self.context_buffer[-1]
            summary += f"Latest: User: {last['user']}\nBot: {last['bot']}"
        
        return summary
    
    def _get_recent_context(self) -> str:
        """Return only recent turns (last 3)"""
        recent_turns = list(self.context_buffer)[-3:]
        lines = []
        
        for turn in recent_turns:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Bot: {turn['bot']}")
        
        return "\n".join(lines)
    
    def get_context_relevance(self, query: str) -> float:
        """
        Score how relevant the context is to new query (0.0 to 1.0)
        
        Low score = context is stale, might need reset
        High score = context is relevant
        """
        if not self.context_buffer:
            return 0.0
        
        # Simple relevance: shared words between query and history
        query_words = set(query.lower().split())
        history_words = set()
        
        for turn in self.context_buffer:
            history_words.update(turn['user'].lower().split())
            history_words.update(turn['bot'].lower().split())
        
        shared = len(query_words & history_words)
        total = len(query_words | history_words)
        
        relevance = shared / total if total > 0 else 0.0
        
        # Time decay: older context is less relevant
        age_seconds = (datetime.now() - self.last_update).total_seconds()
        time_decay = max(0.1, 1.0 - (age_seconds / 3600))  # Decay over 1 hour
        
        return relevance * time_decay
    
    def should_reset(self, relevance_threshold: float = 0.2) -> bool:
        """Determine if context should be reset"""
        return self.get_context_relevance("") < relevance_threshold
    
    def get_context_stats(self) -> Dict:
        """Get statistics about current context"""
        return {
            'turns': len(self.context_buffer),
            'max_turns': self.max_turns,
            'last_update': self.last_update.isoformat(),
            'memory_type': self.memory_type,
            'total_words': sum(
                len(turn['user'].split()) + len(turn['bot'].split())
                for turn in self.context_buffer
            )
        }
    
    def clear(self) -> None:
        """Clear all context"""
        self.context_buffer.clear()
        self.context_summary = ""
        self.last_update = datetime.now()


class ContextManager:
    """Global context manager for multiple sessions"""
    
    def __init__(self, db_path: str = 'backend/conversations.db'):
        self.db_path = db_path
        self.sessions = {}  # In-memory cache
    
    def get_or_create_context(
        self,
        session_id: str,
        max_turns: int = 10,
        memory_type: str = 'full'
    ) -> ConversationContext:
        """Get or create context for a session"""
        if session_id not in self.sessions:
            context = ConversationContext(session_id, max_turns, memory_type)
            # Load from database
            self._load_context_from_db(context)
            self.sessions[session_id] = context
        
        return self.sessions[session_id]
    
    def _load_context_from_db(self, context: ConversationContext) -> None:
        """Load context history from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                SELECT user_input, bot_response FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (context.session_id, context.max_turns))
            
            rows = c.fetchall()
            conn.close()
            
            # Add in chronological order
            for user_input, bot_response in reversed(rows):
                context.add_turn(user_input, bot_response)
        
        except Exception as e:
            print(f"Error loading context: {str(e)}")
    
    def save_context(self, session_id: str) -> None:
        """Save context to database (already done in main backend, but useful)"""
        if session_id in self.sessions:
            context = self.sessions[session_id]
            # Database save is handled in main app.py
            pass
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary of a session"""
        context = self.get_or_create_context(session_id)
        return context.get_context_stats()
    
    def cleanup_old_sessions(self, hours: int = 24) -> int:
        """Remove sessions not updated in N hours"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, context in self.sessions.items():
            age = (current_time - context.last_update).total_seconds() / 3600
            if age > hours:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        return len(sessions_to_remove)
