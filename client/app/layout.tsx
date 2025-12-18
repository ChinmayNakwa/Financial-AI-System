import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/navbar";
const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "FinAI Assistant",
  description: "Advanced Financial RAG Agent",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-[#020617] text-slate-50 antialiased`}>
        <Navbar />
        {children}
      </body>
    </html>
  );
}