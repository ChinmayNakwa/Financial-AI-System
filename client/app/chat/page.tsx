"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage } from "@/components/chat-message";
import { SendHorizonal, Loader2, Key, ShieldAlert, Info } from "lucide-react";

type Message = { role: "user" | "ai"; content: string };

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Load API Key from localStorage on mount
  useEffect(() => {
    const savedKey = localStorage.getItem("gemini_api_key");
    if (savedKey) setApiKey(savedKey);
  }, []);

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(scrollToBottom, [messages, isLoading]);

  const handleApiKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setApiKey(value);
    localStorage.setItem("gemini_api_key", value);
  };

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    if (!apiKey.trim()) {
      setMessages((prev) => [
        ...prev,
        { role: "ai", content: "⚠️ **Action Required:** Please enter your Gemini API Key in the configuration bar above to use the assistant." },
      ]);
      return;
    }

    const userQuery = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userQuery }]);
    setIsLoading(true);

    try {
      const response = await fetch(process.env.NEXT_PUBLIC_API_URL || "/backend/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          query: userQuery,
          api_key: apiKey 
        }),
      });

      const data = await response.json();
      
      if (!response.ok) {
         const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server Error (${response.status})`);
      }

      
      setMessages((prev) => [...prev, { role: "ai", content: data.answer }]);
    } catch (error: any) {
      setMessages((prev) => [
        ...prev, 
        { role: "ai", content: `❌ **Connection Error:** ${error.message || "Could not reach the backend server."}` }
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex h-[calc(100vh-64px)] flex-col bg-[#020617]">
      {/* API Key Configuration Bar */}
      <div className="border-b border-slate-800 bg-slate-900/40 p-2 backdrop-blur-md">
        <div className="mx-auto flex max-w-3xl items-center gap-3">
          <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            <Key className="h-3.5 w-3.5 text-blue-500" />
            <span>Gemini Key</span>
          </div>
          <Input
            type="password"
            value={apiKey}
            onChange={handleApiKeyChange}
            placeholder="Paste your Gemini API key (e.g. AIza...)"
            className="h-8 flex-1 border-slate-700 bg-slate-950 text-xs text-slate-200 focus-visible:ring-blue-500"
          />
          {!apiKey && (
            <span className="flex items-center gap-1 text-[10px] font-bold text-amber-500 animate-pulse">
              <ShieldAlert className="h-3 w-3" /> REQUIRED
            </span>
          )}
        </div>
      </div>

      <ScrollArea className="flex-1 p-4 lg:p-8">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.length === 0 && (
            <div className="py-20 text-center">
              <div className="mb-6 inline-flex h-16 w-16 items-center justify-center rounded-full bg-blue-500/10">
                <Info className="h-8 w-8 text-blue-500" />
              </div>
              <h2 className="text-2xl font-bold text-slate-200">Financial RAG Agent</h2>
              <p className="mx-auto mt-2 max-w-md text-slate-400">
                Ask complex questions about stocks, US economic indicators (FRED), or crypto markets.
              </p>
              <div className="mt-8 grid grid-cols-1 gap-3 text-left sm:grid-cols-2">
                <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3 text-xs text-slate-400">
                  "What is the current US unemployment rate and how has it changed?"
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3 text-xs text-slate-400">
                  "Compare Apple and Nvidia's P/E ratios and show recent news."
                </div>
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <ChatMessage key={i} role={m.role} content={m.content} />
          ))}
          {isLoading && (
            <div className="flex items-center gap-3 p-6 text-slate-400">
              <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
              <span className="text-sm font-medium animate-pulse">Agent reasoning and retrieving multi-source data...</span>
            </div>
          )}
          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      <div className="border-t border-slate-800 bg-slate-900/50 p-4 backdrop-blur-lg">
        <form onSubmit={handleSubmit} className="mx-auto flex max-w-3xl gap-3">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={apiKey ? "Analyze markets, search FRED data..." : "Please enter your API key above..."}
            className="border-slate-700 bg-slate-950 text-slate-200 focus-visible:ring-blue-500"
            disabled={isLoading || !apiKey}
          />
          <Button 
            type="submit" 
            disabled={isLoading || !input.trim() || !apiKey} 
            className="bg-blue-600 hover:bg-blue-700 transition-colors"
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendHorizonal className="h-4 w-4" />}
          </Button>
        </form>
        <p className="mt-2 text-center text-[10px] text-slate-600">
          Powered by Adaptive, Self, and Corrective RAG Architecture.
        </p>
      </div>
    </div>
  );
}