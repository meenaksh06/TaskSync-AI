import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';

const API_BASE = 'http://localhost:8000';

function App() {
  const [userId] = useState(() => 'user_' + Math.random().toString(36).substr(2, 9));
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(err => console.error('Health check failed:', err));
  }, []);

  return (
    <div className="h-screen w-screen bg-neutral-950 flex flex-col">
      <ChatInterface 
        userId={userId} 
        apiBase={API_BASE}
        health={health}
      />
    </div>
  );
}

export default App;
