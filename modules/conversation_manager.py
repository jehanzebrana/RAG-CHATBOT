"""
Module: conversation_manager.py
Task 3: Conversation Context Management

This module manages conversation history with:
- Session-based storage
- Sliding window (last N messages)
- Thread-safe operations
- Automatic cleanup
- Serialization support

Features:
- Per-session conversation tracking
- Configurable history length
- Memory-efficient storage
- Export/import capabilities
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import threading
from loguru import logger


class ConversationTurn:
    """Represents a single question-answer turn."""
    
    def __init__(self, question: str, answer: str, timestamp: Optional[datetime] = None):
        self.question = question
        self.answer = answer
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationTurn':
        """Create from dictionary."""
        return cls(
            question=data["question"],
            answer=data["answer"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    def __repr__(self) -> str:
        return f"ConversationTurn(Q: {self.question[:50]}..., A: {self.answer[:50]}...)"


class ConversationSession:
    """Manages conversation history for a single session."""
    
    def __init__(self, session_id: str, max_history: int = 10):
        """
        Initialize conversation session.
        
        Args:
            session_id: Unique identifier for this session
            max_history: Maximum number of Q&A pairs to keep
        """
        self.session_id = session_id
        self.max_history = max_history
        self.history: deque[ConversationTurn] = deque(maxlen=max_history)
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        self._lock = threading.Lock()
    
    def add_turn(self, question: str, answer: str) -> None:
        """
        Add a Q&A turn to the conversation history.
        
        Args:
            question: User's question
            answer: System's answer
        """
        with self._lock:
            turn = ConversationTurn(question, answer)
            self.history.append(turn)
            self.last_updated = datetime.utcnow()
            
            logger.debug(
                f"Added turn to session {self.session_id}. "
                f"History size: {len(self.history)}/{self.max_history}"
            )
    
    def get_history(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """
        Get conversation history.
        
        Args:
            n: Number of recent turns to get (None = all)
        
        Returns:
            List of conversation turns (oldest first)
        """
        with self._lock:
            if n is None:
                return list(self.history)
            else:
                # Get last N items
                start_idx = max(0, len(self.history) - n)
                return list(self.history)[start_idx:]
    
    def get_history_as_messages(
        self,
        n: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get history formatted for LLM (as message list).
        
        Args:
            n: Number of recent turns to get
        
        Returns:
            List of {"role": "user/assistant", "content": "..."} dicts
        """
        history = self.get_history(n)
        messages = []
        
        for turn in history:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        
        return messages
    
    def clear_history(self) -> None:
        """Clear all conversation history."""
        with self._lock:
            self.history.clear()
            self.last_updated = datetime.utcnow()
            logger.info(f"Cleared history for session {self.session_id}")
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary for serialization."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "max_history": self.max_history,
                "created_at": self.created_at.isoformat(),
                "last_updated": self.last_updated.isoformat(),
                "history": [turn.to_dict() for turn in self.history]
            }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationSession':
        """Create session from dictionary."""
        session = cls(
            session_id=data["session_id"],
            max_history=data["max_history"]
        )
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_updated = datetime.fromisoformat(data["last_updated"])
        
        for turn_data in data["history"]:
            turn = ConversationTurn.from_dict(turn_data)
            session.history.append(turn)
        
        return session
    
    def get_info(self) -> Dict:
        """Get session metadata."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "num_turns": len(self.history),
                "max_history": self.max_history,
                "created_at": self.created_at.isoformat(),
                "last_updated": self.last_updated.isoformat(),
                "age_minutes": (datetime.utcnow() - self.created_at).total_seconds() / 60
            }


class ConversationManager:
    """
    Manages multiple conversation sessions.
    
    Features:
    - Thread-safe session management
    - Automatic cleanup of old sessions
    - Session statistics
    """
    
    def __init__(
        self,
        max_history_per_session: int = 10,
        session_timeout_hours: int = 24
    ):
        """
        Initialize conversation manager.
        
        Args:
            max_history_per_session: Max Q&A pairs per session
            session_timeout_hours: Hours before inactive session cleanup
        """
        self.max_history = max_history_per_session
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.sessions: Dict[str, ConversationSession] = {}
        self._lock = threading.Lock()
        
        logger.info(
            f"✓ Conversation Manager initialized "
            f"(max_history={max_history_per_session}, "
            f"timeout={session_timeout_hours}h)"
        )
    
    def get_session(self, session_id: str) -> ConversationSession:
        """
        Get or create a conversation session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            ConversationSession instance
        """
        with self._lock:
            if session_id not in self.sessions:
                session = ConversationSession(
                    session_id=session_id,
                    max_history=self.max_history
                )
                self.sessions[session_id] = session
                logger.info(f"Created new session: {session_id}")
            
            return self.sessions[session_id]
    
    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """
        Add a Q&A turn to a session.
        
        Args:
            session_id: Session identifier
            question: User's question
            answer: System's answer
        """
        session = self.get_session(session_id)
        session.add_turn(question, answer)
    
    def get_history(
        self,
        session_id: str,
        n: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            n: Number of recent turns (None = all)
        
        Returns:
            List of conversation turns
        """
        session = self.get_session(session_id)
        return session.get_history(n)
    
    def get_history_as_messages(
        self,
        session_id: str,
        n: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get history formatted for LLM."""
        session = self.get_session(session_id)
        return session.get_history_as_messages(n)
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear history for a specific session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if session existed and was cleared
        """
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].clear_history()
                return True
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session entirely.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if session existed and was deleted
        """
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            return False
    
    def cleanup_old_sessions(self) -> int:
        """
        Remove sessions that haven't been updated recently.
        
        Returns:
            Number of sessions deleted
        """
        with self._lock:
            now = datetime.utcnow()
            to_delete = []
            
            for session_id, session in self.sessions.items():
                age = now - session.last_updated
                if age > self.session_timeout:
                    to_delete.append(session_id)
            
            for session_id in to_delete:
                del self.sessions[session_id]
            
            if to_delete:
                logger.info(f"Cleaned up {len(to_delete)} old sessions")
            
            return len(to_delete)
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all active session IDs."""
        with self._lock:
            return list(self.sessions.keys())
    
    def get_stats(self) -> Dict:
        """Get overall statistics."""
        with self._lock:
            total_turns = sum(
                len(session.history)
                for session in self.sessions.values()
            )
            
            return {
                "total_sessions": len(self.sessions),
                "total_turns": total_turns,
                "max_history_per_session": self.max_history,
                "session_timeout_hours": self.session_timeout.total_seconds() / 3600
            }
    
    def export_session(self, session_id: str) -> Optional[Dict]:
        """Export a session as dictionary."""
        with self._lock:
            if session_id in self.sessions:
                return self.sessions[session_id].to_dict()
            return None
    
    def import_session(self, session_data: Dict) -> None:
        """Import a session from dictionary."""
        with self._lock:
            session = ConversationSession.from_dict(session_data)
            self.sessions[session.session_id] = session
            logger.info(f"Imported session: {session.session_id}")


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = ConversationManager(max_history_per_session=10)
    
    # Add some conversation turns
    session_id = "test-session-123"
    
    manager.add_turn(session_id, "What is Python?", "Python is a programming language...")
    manager.add_turn(session_id, "What about FastAPI?", "FastAPI is a modern web framework...")
    
    # Get history
    history = manager.get_history(session_id)
    print(f"History: {len(history)} turns")
    
    for i, turn in enumerate(history, 1):
        print(f"\nTurn {i}:")
        print(f"Q: {turn.question}")
        print(f"A: {turn.answer}")
    
    # Get stats
    print(f"\nStats: {manager.get_stats()}")
