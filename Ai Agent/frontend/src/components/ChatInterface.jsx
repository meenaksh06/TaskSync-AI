import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Mic, Square, Loader2 } from "lucide-react";
import Message from "./Message";
import WelcomeScreen from "./WelcomeScreen";

const ChatInterface = ({ userId, apiBase, health }) => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const durationRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    const handleQuickAction = (e) => {
      setInputText(e.detail);
      inputRef.current?.focus();
    };
    window.addEventListener("quickAction", handleQuickAction);
    return () => window.removeEventListener("quickAction", handleQuickAction);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const sendMessage = async (text = inputText) => {
    if (!text.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: "user",
      text: text.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText("");
    setIsLoading(true);

    try {
      const response = await fetch(`${apiBase}/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, text: text.trim() }),
      });

      const data = await response.json();

      const assistantMessage = {
        id: Date.now() + 1,
        type: "assistant",
        text: data.response,
        intent: data.intent,
        confidence: data.confidence,
        entities: data.entities,
        state: data.state,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: "assistant",
        text: "Connection error. Please check if the server is running.",
        isError: true,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "audio/mp4",
      });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        clearInterval(durationRef.current);
        setIsLoading(true);
        setIsRecording(false);
        setRecordingDuration(0);

        const mimeType = mediaRecorder.mimeType;
        const blob = new Blob(chunksRef.current, { type: mimeType });

        stream.getTracks().forEach((track) => track.stop());

        const formData = new FormData();
        const extension = mimeType.includes("webm") ? "webm" : "m4a";
        formData.append("audio", blob, `recording.${extension}`);
        formData.append("user_id", userId);

        try {
          const response = await fetch(`${apiBase}/voice`, {
            method: "POST",
            body: formData,
          });

          const data = await response.json();

          if (data.transcription) {
            const userMessage = {
              id: Date.now(),
              type: "user",
              text: data.transcription,
              isVoice: true,
              timestamp: new Date().toISOString(),
            };

            const assistantMessage = {
              id: Date.now() + 1,
              type: "assistant",
              text: data.response,
              intent: data.intent,
              confidence: data.confidence,
              entities: data.entities,
              state: data.state,
              timestamp: new Date().toISOString(),
            };

            setMessages((prev) => [...prev, userMessage, assistantMessage]);
          } else {
            setMessages((prev) => [
              ...prev,
              {
                id: Date.now(),
                type: "assistant",
                text:
                  data.response || "Couldn't process audio. Please try again.",
                timestamp: new Date().toISOString(),
              },
            ]);
          }
        } catch (error) {
          console.error("Voice error:", error);
          setMessages((prev) => [
            ...prev,
            {
              id: Date.now(),
              type: "assistant",
              text: "Error processing voice. Please try again.",
              isError: true,
              timestamp: new Date().toISOString(),
            },
          ]);
        }
        setIsLoading(false);
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingDuration(0);

      durationRef.current = setInterval(() => {
        setRecordingDuration((d) => d + 1);
      }, 1000);
    } catch (error) {
      console.error("Microphone error:", error);
      alert("Could not access microphone. Please check permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
  };

  const formatDuration = (secs) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div className="flex-1 flex flex-col h-full max-w-3xl mx-auto w-full">
      {/* Header */}
      <header className="px-6 py-5 border-b border-neutral-800">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-white tracking-tight">
              TaskFlow AI
            </h1>
            <p className="text-sm text-neutral-500 mt-0.5">
              {health?.google_connected
                ? "Google Calendar connected"
                : "Personal assistant"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full ${
                health?.google_connected
                  ? "bg-neutral-800 text-neutral-300"
                  : "bg-neutral-900 text-neutral-500"
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  health?.google_connected ? "bg-white" : "bg-neutral-600"
                }`}
              />
              {health?.google_connected ? "Connected" : "Offline"}
            </span>
          </div>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-8">
        {messages.length === 0 ? (
          <WelcomeScreen onPromptClick={sendMessage} />
        ) : (
          <div className="space-y-6">
            <AnimatePresence initial={false}>
              {messages.map((message) => (
                <Message key={message.id} message={message} />
              ))}
            </AnimatePresence>

            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex gap-3"
              >
                <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center flex-shrink-0">
                  <span className="text-black text-xs font-semibold">AI</span>
                </div>
                <div className="px-4 py-3 rounded-2xl bg-neutral-900 border border-neutral-800">
                  <div className="flex gap-1">
                    {[0, 1, 2].map((i) => (
                      <motion.div
                        key={i}
                        className="w-2 h-2 rounded-full bg-neutral-500"
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{
                          duration: 1,
                          repeat: Infinity,
                          delay: i * 0.2,
                        }}
                      />
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="px-6 py-4 border-t border-neutral-800">
        <div className="flex items-center gap-3">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isLoading && !isRecording}
            className={`w-10 h-10 rounded-full flex items-center justify-center transition-all ${
              isRecording
                ? "bg-white text-black"
                : "bg-neutral-900 text-neutral-400 hover:text-white hover:bg-neutral-800"
            }`}
          >
            {isRecording ? (
              <Square size={16} fill="currentColor" />
            ) : (
              <Mic size={18} />
            )}
          </button>

          {isRecording ? (
            <div className="flex-1 flex items-center gap-3 px-4 py-2.5 rounded-full bg-neutral-900 border border-neutral-700">
              <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
              <span className="text-sm text-white font-mono">
                {formatDuration(recordingDuration)}
              </span>
              <span className="text-sm text-neutral-500">Recording...</span>
            </div>
          ) : (
            <div className="flex-1 flex items-center gap-2 px-4 py-2.5 rounded-full bg-neutral-900 border border-neutral-800 focus-within:border-neutral-600 transition-colors">
              <input
                ref={inputRef}
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type a message..."
                disabled={isLoading}
                className="flex-1 bg-transparent border-none outline-none text-white text-sm placeholder:text-neutral-600"
              />
            </div>
          )}

          <button
            onClick={() => sendMessage()}
            disabled={!inputText.trim() || isLoading || isRecording}
            className={`w-10 h-10 rounded-full flex items-center justify-center transition-all ${
              inputText.trim() && !isLoading && !isRecording
                ? "bg-white text-black hover:bg-neutral-200"
                : "bg-neutral-900 text-neutral-600 cursor-not-allowed"
            }`}
          >
            {isLoading ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Send size={16} />
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
