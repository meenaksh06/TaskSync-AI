import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ChevronDown, ChevronUp, Mic,
  Calendar, Clock, Check, X, AlertTriangle, 
  Bell, FileText, Link, Users, Mail, 
  MessageSquare, List, HelpCircle, Hand, Smile
} from 'lucide-react';

const ICON_MAP = {
  'calendar': Calendar,
  'clock': Clock,
  'check': Check,
  'x': X,
  'alert': AlertTriangle,
  'bell': Bell,
  'file-text': FileText,
  'link': Link,
  'users': Users,
  'mail': Mail,
  'message-square': MessageSquare,
  'list': List,
  'help-circle': HelpCircle,
  'hand': Hand,
  'smile': Smile,
};

const parseTextWithIcons = (text) => {
  if (!text) return null;
  
  const parts = [];
  const regex = /\{icon:([a-z-]+)\}/g;
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(
        <span key={key++}>{text.slice(lastIndex, match.index)}</span>
      );
    }
    
    const iconName = match[1];
    const IconComponent = ICON_MAP[iconName];
    if (IconComponent) {
      parts.push(
        <IconComponent 
          key={key++} 
          size={14} 
          className="inline-block mx-0.5 align-text-bottom text-neutral-400" 
        />
      );
    }
    
    lastIndex = regex.lastIndex;
  }
  
  if (lastIndex < text.length) {
    parts.push(<span key={key++}>{text.slice(lastIndex)}</span>);
  }
  
  return parts.length > 0 ? parts : text;
};

const renderFormattedText = (text) => {
  if (!text) return null;
  
  const lines = text.split('\n');
  
  return lines.map((line, lineIndex) => {
    const boldRegex = /\*\*([^*]+)\*\*/g;
    let parts = [];
    let lastIdx = 0;
    let match;
    let partKey = 0;
    
    while ((match = boldRegex.exec(line)) !== null) {
      if (match.index > lastIdx) {
        parts.push(
          <span key={`${lineIndex}-${partKey++}`}>
            {parseTextWithIcons(line.slice(lastIdx, match.index))}
          </span>
        );
      }
      parts.push(
        <strong key={`${lineIndex}-${partKey++}`} className="font-semibold text-white">
          {parseTextWithIcons(match[1])}
        </strong>
      );
      lastIdx = boldRegex.lastIndex;
    }
    
    if (lastIdx < line.length) {
      parts.push(
        <span key={`${lineIndex}-${partKey++}`}>
          {parseTextWithIcons(line.slice(lastIdx))}
        </span>
      );
    }
    
    if (parts.length === 0) {
      parts = [<span key={`${lineIndex}-0`}>{parseTextWithIcons(line)}</span>];
    }
    
    return (
      <React.Fragment key={lineIndex}>
        {parts}
        {lineIndex < lines.length - 1 && <br />}
      </React.Fragment>
    );
  });
};

const Message = ({ message }) => {
  const [showMeta, setShowMeta] = useState(false);
  const isUser = message.type === 'user';

  const hasEntities = message.entities && 
    Object.values(message.entities).some(v => v !== null && (Array.isArray(v) ? v.length > 0 : true));

  const formatTime = (ts) => {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}
    >
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
        isUser 
          ? 'bg-neutral-800 text-neutral-400' 
          : 'bg-white text-black'
      }`}>
        {isUser ? (
          <span className="text-xs font-medium">You</span>
        ) : (
          <span className="text-xs font-semibold">AI</span>
        )}
      </div>

      <div className={`flex flex-col max-w-[75%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div className={`px-4 py-3 rounded-2xl ${
          isUser 
            ? 'bg-white text-black rounded-tr-md' 
            : message.isError
              ? 'bg-neutral-900 border border-red-900/50 text-neutral-300 rounded-tl-md'
              : 'bg-neutral-900 border border-neutral-800 text-neutral-300 rounded-tl-md'
        }`}>
          {message.isVoice && isUser && (
            <div className="flex items-center gap-1.5 mb-2 text-neutral-500">
              <Mic size={12} />
              <span className="text-xs">Voice message</span>
            </div>
          )}
          
          <div className="text-sm leading-relaxed">
            {renderFormattedText(message.text)}
          </div>

          {!isUser && (message.intent || hasEntities) && (
            <div className="mt-3 pt-3 border-t border-neutral-800">
              <button
                onClick={() => setShowMeta(!showMeta)}
                className="flex items-center gap-2 text-xs text-neutral-500 hover:text-neutral-300 transition-colors"
              >
                <span className="font-medium">{message.intent?.replace(/_/g, ' ')}</span>
                {message.confidence && (
                  <span className="text-neutral-600">
                    {Math.round(message.confidence * 100)}%
                  </span>
                )}
                {hasEntities && (
                  showMeta ? <ChevronUp size={14} /> : <ChevronDown size={14} />
                )}
              </button>

              {showMeta && hasEntities && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  className="mt-3 space-y-2 overflow-hidden"
                >
                  {message.entities.names?.length > 0 && (
                    <MetaRow icon={Users} label="Names" value={message.entities.names.join(', ')} />
                  )}
                  {message.entities.emails?.length > 0 && (
                    <MetaRow icon={Mail} label="Emails" value={message.entities.emails.join(', ')} />
                  )}
                  {message.entities.datetime && (
                    <MetaRow icon={Clock} label="Time" value={new Date(message.entities.datetime).toLocaleString()} />
                  )}
                  {message.entities.phones?.length > 0 && (
                    <MetaRow icon={MessageSquare} label="Phone" value={message.entities.phones.join(', ')} />
                  )}
                  {message.entities.locations?.length > 0 && (
                    <MetaRow icon={Calendar} label="Location" value={message.entities.locations.join(', ')} />
                  )}
                  {message.entities.organizations?.length > 0 && (
                    <MetaRow icon={FileText} label="Org" value={message.entities.organizations.join(', ')} />
                  )}
                </motion.div>
              )}

              {message.state && message.state !== 'idle' && (
                <div className="mt-2 flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                  <span className="text-xs text-neutral-500">
                    Waiting for {message.state.replace('awaiting_', '').replace(/_/g, ' ')}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        <span className="text-[11px] text-neutral-600 mt-1.5 px-1">
          {formatTime(message.timestamp)}
        </span>
      </div>
    </motion.div>
  );
};

const MetaRow = ({ icon: Icon, label, value }) => (
  <div className="flex items-start gap-2 text-xs">
    <Icon size={12} className="text-neutral-500 mt-0.5 flex-shrink-0" />
    <span className="text-neutral-600 w-14 flex-shrink-0">{label}</span>
    <span className="text-neutral-400">{value}</span>
  </div>
);

export default Message;
