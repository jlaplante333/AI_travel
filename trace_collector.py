import time
import uuid
import json
import threading
import functools
import inspect
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Callable, Union, TypeVar, cast
from contextlib import contextmanager
from datetime import datetime
from flask import Flask, request, g, jsonify, Response
from flask_socketio import SocketIO

# Type definitions
AgentType = str  # 'ollama', 'location', 'openai', 'google-tts', 'lumaai', 'orchestrator'
TraceStatus = str  # 'success', 'error', 'running', 'pending'
F = TypeVar('F', bound=Callable[..., Any])

@dataclass
class ReasoningStep:
    """Represents a single reasoning step within an agent trace."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent: str = ""
    action: str = ""
    input: str = ""
    output: str = ""
    reasoning: str = ""
    status: str = "pending"
    duration: int = 0
    children: List['ReasoningStep'] = field(default_factory=list)
    
    start_time: float = field(default=0.0, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding internal fields."""
        result = asdict(self)
        result.pop('start_time', None)
        return result
    
    def start(self) -> None:
        """Mark the step as started."""
        self.start_time = time.time()
        self.status = "running"
    
    def end(self, status: str = "success", output: str = "", reasoning: str = "") -> None:
        """Mark the step as completed."""
        if self.start_time > 0:
            self.duration = int((time.time() - self.start_time) * 1000)  # ms
        self.status = status
        if output:
            self.output = output
        if reasoning:
            self.reasoning = reasoning

@dataclass
class AgentTrace:
    """Represents a complete trace for a single agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agentType: str = ""
    status: str = "pending"
    duration: int = 0
    steps: List[ReasoningStep] = field(default_factory=list)
    input: str = ""
    output: str = ""
    
    start_time: float = field(default=0.0, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding internal fields."""
        result = asdict(self)
        result.pop('start_time', None)
        result['steps'] = [step.to_dict() for step in self.steps]
        return result
    
    def start(self, input_data: str = "") -> None:
        """Mark the agent trace as started."""
        self.start_time = time.time()
        self.status = "running"
        if input_data:
            self.input = input_data
    
    def end(self, status: str = "success", output: str = "") -> None:
        """Mark the agent trace as completed."""
        if self.start_time > 0:
            self.duration = int((time.time() - self.start_time) * 1000)  # ms
        self.status = status
        if output:
            self.output = output
    
    def add_step(self, step: ReasoningStep) -> ReasoningStep:
        """Add a reasoning step to this agent trace."""
        self.steps.append(step)
        return step

@dataclass
class TraceSession:
    """Represents a complete trace session, containing multiple agent traces."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    userQuery: str = ""
    status: str = "pending"
    duration: int = 0
    agents: List[AgentTrace] = field(default_factory=list)
    
    start_time: float = field(default=0.0, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding internal fields."""
        result = asdict(self)
        result.pop('start_time', None)
        result['agents'] = [agent.to_dict() for agent in self.agents]
        return result
    
    def start(self, query: str = "") -> None:
        """Mark the session as started."""
        self.start_time = time.time()
        self.status = "running"
        if query:
            self.userQuery = query
    
    def end(self, status: str = "success") -> None:
        """Mark the session as completed."""
        if self.start_time > 0:
            self.duration = int((time.time() - self.start_time) * 1000)  # ms
        self.status = status
    
    def add_agent(self, agent: AgentTrace) -> AgentTrace:
        """Add an agent trace to this session."""
        self.agents.append(agent)
        return agent

class TraceCollector:
    """Collects and manages reasoning traces from AI agents."""
    
    def __init__(self, app: Optional[Flask] = None, socketio: Optional[SocketIO] = None):
        self.app = app
        self.socketio = socketio
        self.sessions: Dict[str, TraceSession] = {}
        self.active_session_id: Optional[str] = None
        self.lock = threading.RLock()
        
        if app is not None:
            self.init_app(app)
        
        if socketio is not None:
            self.init_socketio(socketio)
    
    def init_app(self, app: Flask) -> None:
        """Initialize with a Flask app."""
        self.app = app
        
        # Register before_request and after_request handlers
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        
        # Add route for SSE
        @app.route('/traces/events')
        def trace_events():
            def generate():
                last_id = request.headers.get('Last-Event-ID', '0')
                try:
                    last_id = int(last_id)
                except ValueError:
                    last_id = 0
                
                session_id = request.args.get('session_id', self.active_session_id)
                if not session_id:
                    yield f"data: {json.dumps({'error': 'No active session'})}\n\n"
                    return
                
                # Initial data
                if session_id in self.sessions:
                    yield f"id: {last_id + 1}\n"
                    yield f"data: {json.dumps(self.sessions[session_id].to_dict())}\n\n"
                
                # Keep connection open for real-time updates
                # In a real implementation, this would use a message queue
                # For simplicity, we're just sending periodic updates
                count = last_id + 2
                while True:
                    time.sleep(1)  # Poll every second
                    if session_id in self.sessions:
                        session = self.sessions[session_id]
                        yield f"id: {count}\n"
                        yield f"data: {json.dumps(session.to_dict())}\n\n"
                        count += 1
            
            return Response(generate(), mimetype='text/event-stream')
        
        # Add route to get all sessions
        @app.route('/traces/sessions')
        def get_sessions():
            return jsonify([session.to_dict() for session in self.sessions.values()])
        
        # Add route to get a specific session
        @app.route('/traces/sessions/<session_id>')
        def get_session(session_id):
            if session_id in self.sessions:
                return jsonify(self.sessions[session_id].to_dict())
            return jsonify({'error': 'Session not found'}), 404
    
    def init_socketio(self, socketio: SocketIO) -> None:
        """Initialize with a SocketIO instance for real-time updates."""
        self.socketio = socketio
    
    def before_request(self) -> None:
        """Flask before_request handler to start a trace session."""
        # Skip for trace API endpoints and static files
        if request.path.startswith('/traces') or request.path.startswith('/static'):
            return
        
        # Start a new session for API endpoints
        if request.path.startswith('/'):
            user_query = ''
            if request.is_json:
                data = request.get_json(silent=True) or {}
                user_query = data.get('prompt', '')
            
            with self.lock:
                session = TraceSession()
                session.start(user_query)
                self.sessions[session.id] = session
                self.active_session_id = session.id
                g.trace_session_id = session.id
    
    def after_request(self, response):
        """Flask after_request handler to end a trace session."""
        session_id = getattr(g, 'trace_session_id', None)
        if session_id and session_id in self.sessions:
            with self.lock:
                session = self.sessions[session_id]
                status = 'success' if 200 <= response.status_code < 300 else 'error'
                session.end(status)
                
                # Emit update via SocketIO if available
                if self.socketio:
                    self.socketio.emit('trace_update', session.to_dict())
        
        return response
    
    def get_active_session(self) -> Optional[TraceSession]:
        """Get the currently active trace session."""
        session_id = getattr(g, 'trace_session_id', self.active_session_id)
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        return None
    
    def create_session(self, user_query: str = "") -> TraceSession:
        """Create a new trace session."""
        with self.lock:
            session = TraceSession()
            session.start(user_query)
            self.sessions[session.id] = session
            self.active_session_id = session.id
            g.trace_session_id = session.id
            return session
    
    def get_or_create_session(self) -> TraceSession:
        """Get the active session or create a new one."""
        session = self.get_active_session()
        if not session:
            session = self.create_session()
        return session
    
    def trace_agent(self, agent_type: str) -> Callable[[F], F]:
        """Decorator to trace an entire agent function."""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                session = self.get_or_create_session()
                
                # Extract input from args or kwargs
                input_data = ""
                if len(args) > 0:
                    input_data = str(args[0])
                elif kwargs:
                    input_data = str(kwargs)
                
                # Create agent trace
                agent_trace = AgentTrace(agentType=agent_type)
                agent_trace.start(input_data)
                session.add_agent(agent_trace)
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Record the output
                    output_data = str(result)
                    agent_trace.end("success", output_data)
                    
                    # Emit update via SocketIO if available
                    if self.socketio:
                        self.socketio.emit('agent_update', agent_trace.to_dict())
                    
                    return result
                except Exception as e:
                    # Record the error
                    agent_trace.end("error", str(e))
                    
                    # Emit update via SocketIO if available
                    if self.socketio:
                        self.socketio.emit('agent_update', agent_trace.to_dict())
                    
                    raise
            
            return cast(F, wrapper)
        
        return decorator
    
    @contextmanager
    def trace_step(self, agent_type: str, action: str, input_data: str = "", reasoning: str = ""):
        """Context manager to trace a reasoning step."""
        session = self.get_or_create_session()
        
        # Find the agent trace or create a new one
        agent_trace = None
        for agent in session.agents:
            if agent.agentType == agent_type:
                agent_trace = agent
                break
        
        if not agent_trace:
            agent_trace = AgentTrace(agentType=agent_type)
            agent_trace.start()
            session.add_agent(agent_trace)
        
        # Create the reasoning step
        step = ReasoningStep(
            agent=agent_type,
            action=action,
            input=input_data,
            reasoning=reasoning
        )
        step.start()
        agent_trace.add_step(step)
        
        # Emit update via SocketIO if available
        if self.socketio:
            self.socketio.emit('step_update', step.to_dict())
        
        try:
            # Execute the step
            yield step
            
            # Mark as successful if not already ended
            if step.status == "running":
                step.end("success")
        except Exception as e:
            # Record the error
            step.end("error", str(e))
            raise
        finally:
            # Emit final update via SocketIO if available
            if self.socketio:
                self.socketio.emit('step_update', step.to_dict())
    
    def record_step(self, agent_type: str, action: str, input_data: str, output_data: str, 
                   reasoning: str = "", status: str = "success", duration: int = 0) -> ReasoningStep:
        """Record a completed reasoning step."""
        session = self.get_or_create_session()
        
        # Find the agent trace or create a new one
        agent_trace = None
        for agent in session.agents:
            if agent.agentType == agent_type:
                agent_trace = agent
                break
        
        if not agent_trace:
            agent_trace = AgentTrace(agentType=agent_type)
            agent_trace.start()
            session.add_agent(agent_trace)
        
        # Create the reasoning step
        step = ReasoningStep(
            agent=agent_type,
            action=action,
            input=input_data,
            output=output_data,
            reasoning=reasoning,
            status=status,
            duration=duration
        )
        agent_trace.add_step(step)
        
        # Emit update via SocketIO if available
        if self.socketio:
            self.socketio.emit('step_update', step.to_dict())
        
        return step
    
    def export_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Export a session as a dictionary."""
        if not session_id:
            session_id = self.active_session_id
        
        if session_id and session_id in self.sessions:
            return self.sessions[session_id].to_dict()
        
        return {}
    
    def export_all_sessions(self) -> List[Dict[str, Any]]:
        """Export all sessions as a list of dictionaries."""
        return [session.to_dict() for session in self.sessions.values()]
    
    def clear_sessions(self) -> None:
        """Clear all trace sessions."""
        with self.lock:
            self.sessions.clear()
            self.active_session_id = None
