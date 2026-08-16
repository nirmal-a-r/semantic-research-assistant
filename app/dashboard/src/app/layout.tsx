import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ProjectDataProvider } from "@/lib/ProjectDataContext";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Adaptive Hybrid Semantic Research Assistant",
  description:
    "Retrieval-first, evidence-grounded, explainable QA system combining BM25, Dense FAISS retrieval, RRF fusion, cross-encoder re-ranking, and fine-tuned extractive QA on SQuAD v1.1. Course: 23AID472 Text Analytics, Batch A-17.",
  keywords: ["NLP", "QA", "BM25", "FAISS", "RoBERTa", "SQuAD", "retrieval-augmented"],
  openGraph: {
    title: "Adaptive Hybrid Semantic Research Assistant",
    description: "Retrieval-first, evidence-grounded, explainable QA on SQuAD v1.1",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={inter.variable}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased" style={{ background: "var(--bg-base)" }}>
        <ProjectDataProvider>
          {children}
        </ProjectDataProvider>
      </body>
    </html>
  );
}
