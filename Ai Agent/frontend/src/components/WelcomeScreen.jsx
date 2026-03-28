import React from 'react';
import { motion } from 'framer-motion';
import { Calendar, Bell, Mail, ArrowRight } from 'lucide-react';

const WelcomeScreen = ({ onPromptClick }) => {
  const suggestions = [
    {
      icon: Calendar,
      text: 'Schedule a meeting with John tomorrow at 2pm',
    },
    {
      icon: Bell,
      text: 'Remind me to call the client at 5pm',
    },
    {
      icon: Calendar,
      text: "What's on my calendar today?",
    },
    {
      icon: Mail,
      text: 'Send meeting invite to team@company.com',
    },
  ];

  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="text-center mb-12"
      >
        <div className="w-25 h-16 rounded-2xl bg-white flex items-center justify-center mx-auto mb-6">
          <span className="text-black text-xl font-bold">One prompt solution to all workflows!</span>
        </div>

        <h1 className="text-2xl font-semibold text-white mb-3">
          How can I help you?
        </h1>
        
        <p className="text-neutral-500 max-w-md">
          I can schedule meetings, set reminders, manage your calendar, and connect with Google services.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.4 }}
        className="w-full max-w-lg space-y-2"
      >
        {suggestions.map((item, index) => (
          <motion.button
            key={index}
            onClick={() => onPromptClick(item.text)}
            className="w-full flex items-center gap-4 px-4 py-3.5 rounded-xl bg-neutral-900 border border-neutral-800 hover:border-neutral-700 hover:bg-neutral-800/50 transition-all text-left group"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * index + 0.3 }}
          >
            <div className="w-9 h-9 rounded-lg bg-neutral-800 flex items-center justify-center flex-shrink-0 group-hover:bg-neutral-700 transition-colors">
              <item.icon size={18} className="text-neutral-400" />
            </div>
            <span className="flex-1 text-sm text-neutral-300 group-hover:text-white transition-colors">
              {item.text}
            </span>
            <ArrowRight size={16} className="text-neutral-600 group-hover:text-neutral-400 transition-colors" />
          </motion.button>
        ))}
      </motion.div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="mt-8 text-xs text-neutral-600"
      >
        Press Enter to send · Click mic for voice input
      </motion.p>
    </div>
  );
};

export default WelcomeScreen;
