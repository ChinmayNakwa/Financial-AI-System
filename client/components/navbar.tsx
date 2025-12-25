import Link from "next/link";
import { FaGithub } from 'react-icons/fa'

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 border-b border-slate-800 bg-[#020617]/80 backdrop-blur-md">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <Link href="/" className="text-xl font-bold tracking-tight text-blue-500">
          FinAI<span className="text-slate-100">Assistant</span>
        </Link>
        <div className="flex gap-6 text-sm font-medium">
          <Link href="/" className="transition-colors hover:text-blue-400">About</Link>
          <Link href="/chat" className="transition-colors hover:text-blue-400">Chat</Link>
          <Link href="https://github.com/ChinmayNakwa/Financial-AI-System" target="_blank" className="transition-colors hover:text-blue-400">  <FaGithub className="h-5 w-5" /></Link>
        </div>
      </div>
    </nav>
  );
}