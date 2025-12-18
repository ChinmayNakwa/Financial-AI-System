import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Avatar } from "@/components/ui/avatar";
import { User, Bot } from "lucide-react";

export function ChatMessage({ role, content }: { role: "user" | "ai"; content: string }) {
  return (
    <div className={`flex w-full gap-4 py-6 ${role === "ai" ? "bg-slate-900/30 px-4 rounded-xl" : "px-4"}`}>
      <Avatar className="h-8 w-8 border border-slate-700 bg-slate-800 flex items-center justify-center">
        {role === "user" ? <User size={18} /> : <Bot size={18} className="text-blue-500" />}
      </Avatar>
      <div className="flex-1 space-y-2 overflow-hidden">
        <p className="text-xs font-bold uppercase tracking-wider text-slate-500">
          {role === "user" ? "You" : "Financial Assistant"}
        </p>
        <div className="prose prose-invert max-w-none text-slate-300">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}