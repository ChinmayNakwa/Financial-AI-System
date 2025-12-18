import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { BrainCircuit, ShieldCheck, Zap } from "lucide-react";

export default function AboutPage() {
  const features = [
    {
      title: "Adaptive RAG",
      desc: "Intelligently routes queries to the best source—Yahoo Finance, FRED, or SEC filings.",
      icon: <Zap className="h-8 w-8 text-blue-500" />,
    },
    {
      title: "Self-RAG",
      desc: "Quality analyst node that verifies retrieved data for relevance and accuracy.",
      icon: <ShieldCheck className="h-8 w-8 text-emerald-500" />,
    },
    {
      title: "Corrective RAG",
      desc: "Automated multi-step research to fill knowledge gaps and reconcile data.",
      icon: <BrainCircuit className="h-8 w-8 text-purple-500" />,
    },
  ];

  return (
    <main className="container mx-auto px-4 py-20 text-center">
      <h1 className="mb-6 text-5xl font-extrabold tracking-tight lg:text-6xl">
        The Future of <span className="text-blue-500">Financial Intelligence.</span>
      </h1>
      <p className="mx-auto mb-10 max-w-2xl text-lg text-slate-400">
        An advanced agentic system built with LangGraph to provide deep, 
        multi-source financial analysis with self-correcting logic.
      </p>
      
      <Link href="/chat">
        <Button size="lg" className="bg-blue-600 hover:bg-blue-700">
          Start Researching
        </Button>
      </Link>

      <div className="mt-24 grid gap-8 md:grid-cols-3">
        {features.map((f, i) => (
          <Card key={i} className="border-slate-800 bg-slate-900/50 backdrop-blur-sm">
            <CardHeader>
              <div className="mb-4 flex justify-center">{f.icon}</div>
              <CardTitle>{f.title}</CardTitle>
            </CardHeader>
            <CardContent className="text-slate-400">{f.desc}</CardContent>
          </Card>
        ))}
      </div>
    </main>
  );
}