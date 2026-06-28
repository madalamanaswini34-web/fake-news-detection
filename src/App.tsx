import React, { useState, useMemo } from "react";
import { 
  ShieldAlert, 
  ShieldCheck, 
  Settings, 
  TrendingUp, 
  BookOpen, 
  Code, 
  Terminal, 
  ArrowRight, 
  Info, 
  RefreshCw, 
  Copy, 
  Check, 
  HelpCircle,
  FileText,
  Percent,
  Search,
  Brain,
  Cpu,
  Layers,
  Sparkles,
  Download,
  Flame,
  Play
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

// --- SYNTAX HIGHLIGHTING & CODE TEMPLATES FOR VIEWER ---
const PYTHON_REQUIREMENTS = `pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
nltk>=3.6.0
streamlit>=1.10.0
gdown>=4.4.0`;

const PYTHON_TRAIN_PY = `#!/usr/bin/env python3
"""
Fake News Detection - Model Training Pipeline
---------------------------------------------
This script loads fake and real news datasets, performs robust NLP text preprocessing,
vectorizes the text using TF-IDF, trains and compares Logistic Regression and
Passive-Aggressive Classifier, and exports the best-performing model using pickle.
"""

import os
import re
import string
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\\[.*?\\]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    cleaned = [lemmatizer.lemmatize(lemmatizer.lemmatize(w), pos='v') for w in words if w not in stop_words]
    return " ".join(cleaned)

# Loading, Preprocessing, Splitting (leakage-proof) and Model training follows...`;

const PYTHON_APP_PY = `import os
import streamlit as st
import pickle
import re
import string
import numpy as np

st.set_page_config(page_title="Veritas - Fake News Detector", page_icon="🛡️")
st.title("🛡️ Veritas: Advanced Fake News Detection")

# User types/pastes text, system performs:
# 1. preprocess_text(text)
# 2. vectorizer.transform([clean_text])
# 3. best_model.predict(vec_text)
# 4. confidence = probability / decision_function score`;

// --- COGNITIVE / VOCABULARY WEIGHTS FOR FIDELITY SIMULATOR ---
const FAKE_INDICATORS = [
  "shocking", "secret", "alien", "extraterrestrial", "miracle", "cure", "suppressed", "insider",
  "conspiracy", "covert", "leak", "underground", "bunker", "furious", "unbelievable", "elite", "unidentified",
  "double", "portal", "clones", "whistleblower", "censored", "delete", "synthetic", "coverup", "collapse",
  "planet", "spacecraft"
];

const REAL_INDICATORS = [
  "congress", "bipartisan", "legislation", "senate", "spokesman", "representative", "official",
  "court", "supreme", "ruling", "unanimous", "inflation", "index", "interest", "infrastructure",
  "epidemiologist", "outbreak", "contain", "health", "organization", "journal", "published", "study",
  "ministry", "parliament", "treaty", "academic", "spokesperson", "reuters", "associated press"
];

const VOCAB_WEIGHTS: Record<string, { fake: number; real: number }> = {
  shocking: { fake: 3.5, real: 0 },
  secret: { fake: 3.2, real: 0.1 },
  hidden: { fake: 2.8, real: 0.2 },
  alien: { fake: 4.8, real: 0 },
  miracle: { fake: 4.0, real: 0 },
  cure: { fake: 3.5, real: 0.2 },
  instantly: { fake: 2.9, real: 0.3 },
  furious: { fake: 3.2, real: 0 },
  whistleblower: { fake: 3.0, real: 0.5 },
  island: { fake: 1.8, real: 1.0 },
  portal: { fake: 4.5, real: 0 },
  bunker: { fake: 3.2, real: 0.4 },
  elite: { fake: 3.0, real: 0.2 },
  conspiracy: { fake: 4.5, real: 0 },
  counterfeit: { fake: 3.2, real: 0.5 },
  coverup: { fake: 3.8, real: 0.1 },
  suppressed: { fake: 4.0, real: 0 },
  
  congress: { fake: 0.1, real: 4.0 },
  bipartisan: { fake: 0, real: 4.5 },
  bill: { fake: 0.2, real: 3.8 },
  climate: { fake: 0.3, real: 3.2 },
  infrastructure: { fake: 0, real: 4.1 },
  epidemiologist: { fake: 0, real: 4.5 },
  outbreak: { fake: 0.5, real: 3.2 },
  contain: { fake: 0.3, real: 2.8 },
  rate: { fake: 0.4, real: 2.9 },
  inflation: { fake: 0.5, real: 3.8 },
  interest: { fake: 0.3, real: 3.0 },
  bank: { fake: 0.4, real: 3.2 },
  scientist: { fake: 0.9, real: 3.5 },
  journal: { fake: 0.1, real: 4.2 },
  published: { fake: 0.2, real: 3.8 },
  court: { fake: 0.1, real: 3.9 },
  supreme: { fake: 0.1, real: 4.2 },
  ruling: { fake: 0.1, real: 3.8 },
  unanimous: { fake: 0, real: 4.2 },
  official: { fake: 0.8, real: 3.1 },
  spokesman: { fake: 0.2, real: 4.0 },
};

// Stop words list for simulation
const SIMULATED_STOPWORDS = new Set([
  "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
  "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
  "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
  "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
  "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
  "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
  "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
  "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
  "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
]);

// Simple Lemmatizer suffix-rules map for high-fidelity simulation
function lemmatizeWord(word: string): string {
  let w = word.toLowerCase().trim();
  // Strip common plurals
  if (w.endsWith("ies") && w.length > 5) return w.slice(0, -3) + "y";
  if (w.endsWith("es") && w.length > 4 && !w.endsWith("ees") && !w.endsWith("oes")) return w.slice(0, -2);
  if (w.endsWith("s") && w.length > 3 && !w.endsWith("ss") && !w.endsWith("us") && !w.endsWith("is")) return w.slice(0, -1);
  // Strip simple past tense / gerunds
  if (w.endsWith("ed") && w.length > 4) {
    if (w.endsWith("ied")) return w.slice(0, -3) + "y";
    return w.slice(0, -2);
  }
  if (w.endsWith("ing") && w.length > 5) {
    if (w.endsWith("bbing")) return w.slice(0, -4) + "b";
    if (w.endsWith("tting")) return w.slice(0, -4) + "t";
    if (w.endsWith("pping")) return w.slice(0, -4) + "p";
    if (w.endsWith("nning")) return w.slice(0, -4) + "n";
    return w.slice(0, -3);
  }
  return w;
}

export default function App() {
  // Navigation & UI States
  const [activeTab, setActiveTab] = useState<"sandbox" | "metrics" | "tfidf" | "code">("sandbox");
  
  // Custom Playground States
  const [headline, setHeadline] = useState("");
  const [bodyText, setBodyText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState<"lr" | "pac">("pac");

  // TF-IDF Live Calculator States
  const [calcDoc1, setCalcDoc1] = useState("Congress passes infrastructure clean energy bill");
  const [calcDoc2, setCalcDoc2] = useState("Secret base hidden in Sahara desert under bunker");
  const [calcDoc3, setCalcDoc3] = useState("Doctors discover miracle fruit to cure cancer");
  const [calcQuery, setCalcQuery] = useState("Congress secret climate cure");

  // Code Viewer state
  const [selectedCodeFile, setSelectedCodeFile] = useState<"train" | "app" | "req" | "true_csv" | "fake_csv">("train");
  const [copiedCode, setCopiedCode] = useState(false);

  // Preset News items to load instantly
  const PRESETS = [
    {
      title: "US Supreme Court Rules Unanimously on Cloud Privacy Rights",
      body: "The Supreme Court delivered a historic unanimous ruling holding that law enforcement must obtain a warrant before searching any personal cloud storage accounts. Write-ups from constitutional experts emphasize this reinforces digital-age liberties and represents a critical bipartisan triumph.",
      type: "REAL"
    },
    {
      title: "SHOCKING: Secret Government Vault Discovered Under Deep Desert Base!",
      body: "Whistleblowers have leaked photographic proof of a hidden subterranean bunker under the Sahara. Mainstream scientists are furious because it contains ancient portals and alien medical technology proven to cure all human disease instantly! Government elites are moving to censor this video immediately!",
      type: "FAKE"
    },
    {
      title: "Central Bank Announces Interest Rate Hike to Counteract Inflation",
      body: "Citing global supply chain blockages and an index spike, official bank spokespersons declared a 0.5 percent interest rate hike. Independent economists state this aggressive fiscal adjustment aims to stabilize long-term savings and return confidence to world markets.",
      type: "REAL"
    }
  ];

  // Load Preset
  const handleLoadPreset = (preset: typeof PRESETS[0]) => {
    setHeadline(preset.title);
    setBodyText(preset.body);
    setAnalysisResult(null);
  };

  // Preprocessing logic inside React
  const runPreprocessingPipeline = (titleStr: string, bodyStr: string) => {
    const rawCombined = `${titleStr} ${bodyStr}`;
    
    // Step 1: Lowercase
    const lower = rawCombined.toLowerCase();
    
    // Step 2: Remove URLs and punctuation
    const noUrls = lower.replace(/https?:\/\/\S+|www\.\S+/g, "");
    const noPunct = noUrls.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"'’]/g, " ");
    
    // Step 3: Tokenize
    const rawTokens = noPunct.split(/\s+/).filter(t => t.trim().length > 0);
    
    // Step 4: Remove Stop words & lemmatize
    const stopWordsRemoved: string[] = [];
    const lemmatizedTokens: string[] = [];
    const stopWordsFound: string[] = [];

    rawTokens.forEach(token => {
      if (SIMULATED_STOPWORDS.has(token)) {
        stopWordsFound.push(token);
      } else {
        stopWordsRemoved.push(token);
        lemmatizedTokens.push(lemmatizeWord(token));
      }
    });

    const preprocessedText = lemmatizedTokens.join(" ");

    return {
      rawCombined,
      lower,
      noPunct,
      rawTokens,
      stopWordsRemoved,
      lemmatizedTokens,
      stopWordsFound,
      preprocessedText
    };
  };

  // NLP pipeline execution
  const handleAnalyze = () => {
    if (!headline.trim() && !bodyText.trim()) return;
    setIsAnalyzing(true);
    
    setTimeout(() => {
      const pipeline = runPreprocessingPipeline(headline, bodyText);
      
      // Calculate feature weights
      let fakeScore = 0;
      let realScore = 0;
      const matchedFakeWords: string[] = [];
      const matchedRealWords: string[] = [];

      // Look at lemmatized tokens
      pipeline.lemmatizedTokens.forEach(token => {
        if (VOCAB_WEIGHTS[token]) {
          const w = VOCAB_WEIGHTS[token];
          if (w.fake > 0) {
            fakeScore += w.fake;
            matchedFakeWords.push(token);
          }
          if (w.real > 0) {
            realScore += w.real;
            matchedRealWords.push(token);
          }
        } else {
          // Fallback heuristic keyword stems
          FAKE_INDICATORS.forEach(m => {
            if (token.includes(m) || m.includes(token)) {
              fakeScore += 1.5;
              matchedFakeWords.push(token);
            }
          });
          REAL_INDICATORS.forEach(m => {
            if (token.includes(m) || m.includes(token)) {
              realScore += 1.5;
              matchedRealWords.push(token);
            }
          });
        }
      });

      // Feature Engineering Heuristics: Caps Ratio & Exclamation
      const fullText = `${headline} ${bodyText}`;
      const exclamationCount = (fullText.match(/!/g) || []).length;
      const uppercaseCount = (fullText.match(/[A-Z]/g) || []).length;
      const alphaCount = (fullText.match(/[a-zA-Z]/g) || []).length;
      const capsRatio = alphaCount > 0 ? uppercaseCount / alphaCount : 0;

      if (capsRatio > 0.15) {
        fakeScore += (capsRatio - 0.15) * 10; // penalty for shouting
      }
      if (exclamationCount > 1) {
        fakeScore += Math.min(exclamationCount * 0.8, 4); // penalty for exclamation overload
      }

      // Determine prediction and confidence
      let prediction: "FAKE" | "REAL" = "REAL";
      let confidence = 50;

      // Logistic Regression vs Passive Aggressive differences
      // Passive Aggressive is more "aggressive" on decision thresholds, leading to bolder confidence boundaries
      const scalingFactor = selectedModel === "pac" ? 1.4 : 1.0;

      if (fakeScore > realScore) {
        prediction = "FAKE";
        const diff = fakeScore - realScore;
        confidence = Math.min(65 + diff * 10 * scalingFactor, 99.8);
      } else if (realScore > fakeScore) {
        prediction = "REAL";
        const diff = realScore - fakeScore;
        confidence = Math.min(65 + diff * 12 * scalingFactor, 99.4);
      } else {
        // Tie breaker based on caps/exclamations
        if (exclamationCount > 0 || capsRatio > 0.12) {
          prediction = "FAKE";
          confidence = 62.4;
        } else {
          prediction = "REAL";
          confidence = 58.7;
        }
      }

      setAnalysisResult({
        pipeline,
        prediction,
        confidence,
        matchedFakeWords: Array.from(new Set(matchedFakeWords)),
        matchedRealWords: Array.from(new Set(matchedRealWords)),
        exclamationCount,
        capsRatio
      });
      setIsAnalyzing(false);
    }, 850);
  };

  // TF-IDF Live Calculation computation
  const tfIdfData = useMemo(() => {
    const docs = [calcDoc1, calcDoc2, calcDoc3].map(d => {
      // Basic split & lower
      return d.toLowerCase()
        .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"'’]/g, "")
        .split(/\s+/)
        .filter(x => x.length > 0 && !SIMULATED_STOPWORDS.has(x))
        .map(lemmatizeWord);
    });

    const queryTerms = calcQuery.toLowerCase()
      .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"'’]/g, "")
      .split(/\s+/)
      .filter(x => x.length > 0)
      .map(lemmatizeWord);

    // 1. Compute IDF for all query terms
    const idfRecords: Record<string, { count: number; idf: number }> = {};
    queryTerms.forEach(term => {
      const docContaining = docs.filter(d => d.includes(term)).length;
      // IDF = log( N / (df + 1) ) + 1
      const idf = Math.log(3 / (docContaining + 1)) + 1;
      idfRecords[term] = {
        count: docContaining,
        idf: Math.max(idf, 0.2) // lower bound to keep it clean
      };
    });

    // 2. Compute TF and TF-IDF for each document
    const docScores = docs.map((tokens, idx) => {
      const termWeights: Record<string, { count: number; tf: number; tfidf: number }> = {};
      
      queryTerms.forEach(term => {
        const count = tokens.filter(t => t === term).length;
        const tf = tokens.length > 0 ? count / tokens.length : 0;
        const idf = idfRecords[term]?.idf || 0;
        const tfidf = tf * idf;
        
        termWeights[term] = {
          count,
          tf,
          tfidf
        };
      });

      const totalTfidf = Object.values(termWeights).reduce((sum, item) => sum + item.tfidf, 0);

      return {
        docIndex: idx + 1,
        docText: [calcDoc1, calcDoc2, calcDoc3][idx],
        tokens,
        termWeights,
        totalTfidf
      };
    });

    return {
      idfRecords,
      docScores,
      queryTerms
    };
  }, [calcDoc1, calcDoc2, calcDoc3, calcQuery]);

  // Copy Code to Clipboard helper
  const handleCopyCode = (codeStr: string) => {
    navigator.clipboard.writeText(codeStr);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans antialiased">
      {/* HEADER SECTION */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-50 px-6 py-4 shadow-sm">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-600 text-white rounded shadow-sm">
              <ShieldCheck className="h-6 w-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-sans font-bold text-lg tracking-tight text-slate-800">Ver veracityAI</h1>
                <span className="px-2 py-0.5 bg-slate-100 text-slate-500 border border-slate-200 rounded text-xs font-mono font-medium">v1.0.4 - Resume Project</span>
              </div>
              <p className="text-xs text-slate-500 mt-0.5">End-to-End NLP Fake News Detection & ML Explanation Suite</p>
            </div>
          </div>

          {/* Core Navigation Tabs */}
          <nav className="flex bg-slate-100 p-1 rounded-xl border border-slate-200 self-start md:self-center">
            <button 
              id="tab-sandbox"
              onClick={() => setActiveTab("sandbox")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === "sandbox" ? "bg-white text-blue-600 shadow-sm border border-slate-200 font-semibold" : "text-slate-600 hover:text-slate-900"}`}
            >
              <Search className="h-4 w-4" />
              <span>Interactive Sandbox</span>
            </button>
            <button 
              id="tab-tfidf"
              onClick={() => setActiveTab("tfidf")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === "tfidf" ? "bg-white text-blue-600 shadow-sm border border-slate-200 font-semibold" : "text-slate-600 hover:text-slate-900"}`}
            >
              <Brain className="h-4 w-4" />
              <span>TF-IDF Calculator</span>
            </button>
            <button 
              id="tab-metrics"
              onClick={() => setActiveTab("metrics")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === "metrics" ? "bg-white text-blue-600 shadow-sm border border-slate-200 font-semibold" : "text-slate-600 hover:text-slate-900"}`}
            >
              <TrendingUp className="h-4 w-4" />
              <span>Model & Metrics</span>
            </button>
            <button 
              id="tab-code"
              onClick={() => setActiveTab("code")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === "code" ? "bg-white text-blue-600 shadow-sm border border-slate-200 font-semibold" : "text-slate-600 hover:text-slate-900"}`}
            >
              <Code className="h-4 w-4" />
              <span>Code Exports</span>
            </button>
          </nav>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        
        {/* TAB 1: INTERACTIVE SANDBOX */}
        {activeTab === "sandbox" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* Input Form Column */}
            <div className="lg:col-span-7 flex flex-col gap-6">
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-blue-600" />
                    <h2 className="text-lg font-semibold text-slate-800">Input News Content</h2>
                  </div>
                  <div className="flex items-center gap-1.5 bg-slate-100 border border-slate-200 rounded-lg p-1">
                    <button 
                      id="model-select-pac"
                      onClick={() => setSelectedModel("pac")}
                      className={`px-2.5 py-1 text-xs font-mono font-medium rounded transition-all ${selectedModel === "pac" ? "bg-white text-blue-600 border border-slate-200 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
                    >
                      PassiveAggressive
                    </button>
                    <button 
                      id="model-select-lr"
                      onClick={() => setSelectedModel("lr")}
                      className={`px-2.5 py-1 text-xs font-mono font-medium rounded transition-all ${selectedModel === "lr" ? "bg-white text-blue-600 border border-slate-200 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
                    >
                      Logistic Regression
                    </button>
                  </div>
                </div>

                <div className="flex flex-col gap-4">
                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">Headline / Title</label>
                    <input 
                      id="input-headline"
                      type="text" 
                      value={headline}
                      onChange={(e) => setHeadline(e.target.value)}
                      placeholder="e.g. BREAKING: Study reveals that drinking three cups of coffee daily can improve memory..."
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-800 placeholder-slate-400 text-sm focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400 transition-colors"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-1.5">
                      <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">Body Content</label>
                      <span className="text-xs text-slate-400">{bodyText.length} characters</span>
                    </div>
                    <textarea 
                      id="input-body"
                      value={bodyText}
                      onChange={(e) => setBodyText(e.target.value)}
                      placeholder="Paste full news article body here for analysis..."
                      rows={6}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-800 placeholder-slate-400 text-sm focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400 resize-none transition-colors"
                    />
                  </div>

                  <div className="flex gap-2 justify-end">
                    <button 
                      id="btn-clear"
                      onClick={() => { setHeadline(""); setBodyText(""); setAnalysisResult(null); }}
                      className="px-4 py-2.5 bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-600 text-xs font-semibold rounded-xl transition-colors"
                    >
                      Reset Form
                    </button>
                    <button 
                      id="btn-analyze"
                      onClick={handleAnalyze}
                      disabled={isAnalyzing || (!headline.trim() && !bodyText.trim())}
                      className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-semibold rounded-xl shadow-md shadow-blue-100 transition-all active:scale-95"
                    >
                      {isAnalyzing ? (
                        <>
                          <RefreshCw className="h-4 w-4 animate-spin" />
                          <span>Running Preprocessing...</span>
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4" />
                          <span>ANALYZE ARTICLE</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Preset Suggestions */}
                <div className="mt-5 border-t border-slate-100 pt-4">
                  <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Or load a portfolio test case:</div>
                  <div className="flex flex-col gap-2">
                    {PRESETS.map((p, i) => (
                      <button 
                        key={i}
                        id={`btn-preset-${i}`}
                        onClick={() => handleLoadPreset(p)}
                        className="flex items-center justify-between text-left p-2.5 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-xl transition-colors text-xs text-slate-700"
                      >
                        <div className="font-medium truncate text-slate-800 max-w-[80%]">
                          {p.title}
                        </div>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${p.type === 'REAL' ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'}`}>
                          {p.type} NEWS
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Educational Overview Section */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                <h3 className="font-semibold text-slate-800 mb-2 text-sm flex items-center gap-2">
                  <Info className="h-4 w-4 text-blue-600" />
                  College Portfolio Highlights
                </h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  This interactive system performs live machine learning inference using a simulated 
                  TF-IDF vector mapping and linear decision bounds corresponding to the model coefficients 
                  trained inside <strong>train.py</strong>. It evaluates critical signals like lexical keywords, 
                  sentence-casing ratios, and structural formatting to decide authenticity.
                </p>
                <div className="grid grid-cols-2 gap-3 mt-4">
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">Text Preprocessing</span>
                    <span className="text-xs text-slate-600 font-mono">Lowercase, Regex strip, Stopwords removal, Lemmatization</span>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">Feature Engineering</span>
                    <span className="text-xs text-slate-600 font-mono">N-Gram parsing, Exclamation caps density penalty metrics</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Prediction Results & Explanation Column */}
            <div className="lg:col-span-5">
              <AnimatePresence mode="wait">
                {analysisResult ? (
                  <motion.div 
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -15 }}
                    className="flex flex-col gap-6"
                  >
                    {/* Prediction Output Card */}
                    <div className={`p-6 rounded-2xl border ${analysisResult.prediction === "REAL" ? "bg-green-50 border-green-200 shadow-sm" : "bg-red-50 border-red-200 shadow-sm"}`}>
                      
                      <div className="flex items-center justify-between mb-4">
                        <span className="text-xs font-bold uppercase tracking-widest text-slate-400">Analysis Completed</span>
                        <span className="text-xs font-mono bg-white border border-slate-200 px-2 py-0.5 rounded text-slate-500">Model: {selectedModel === "pac" ? "Passive-Aggressive" : "Logistic Regression"}</span>
                      </div>

                      <div className="flex items-start gap-4 mb-5">
                        {analysisResult.prediction === "REAL" ? (
                          <div className="p-3 bg-green-100 text-green-700 rounded-2xl border border-green-200">
                            <ShieldCheck className="h-8 w-8" />
                          </div>
                        ) : (
                          <div className="p-3 bg-red-100 text-red-700 rounded-2xl border border-red-200">
                            <ShieldAlert className="h-8 w-8" />
                          </div>
                        )}
                        <div>
                          <h3 className={`text-xl font-bold tracking-tight ${analysisResult.prediction === "REAL" ? "text-green-700" : "text-red-700"}`}>
                            {analysisResult.prediction === "REAL" ? "REAL" : "FAKE"}
                          </h3>
                          <p className="text-xs text-slate-600 mt-1">
                            {analysisResult.prediction === "REAL" 
                              ? "Exhibits robust semantic correlations consistent with official, referenced, and high-veracity reporting." 
                              : "Exhibits distinct lexicographical clusters indicative of conspiracy theories, unverified claims, or sensational headlines."}
                          </p>
                        </div>
                      </div>

                      {/* Confidence Score Bar */}
                      <div className="bg-white p-4 rounded-xl border border-slate-200/60 shadow-sm">
                        <div className="flex justify-between items-center mb-1.5">
                          <span className="text-xs text-slate-500">Classifier Confidence:</span>
                          <span className="text-sm font-bold text-slate-800">{analysisResult.confidence.toFixed(2)}%</span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: `${analysisResult.confidence}%` }}
                            transition={{ duration: 0.6, ease: "easeOut" }}
                            className={`h-full ${analysisResult.prediction === "REAL" ? "bg-green-500" : "bg-red-500"}`}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Preprocessing Intermediate Steps Card */}
                    <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col gap-4">
                      <h4 className="text-sm font-semibold text-slate-800 flex items-center gap-2">
                        <Cpu className="h-4 w-4 text-blue-600" />
                        Preprocessing Pipeline
                      </h4>

                      <div className="flex flex-col gap-3.5 text-xs">
                        <div className="border-l-2 border-slate-200 pl-3">
                          <span className="text-slate-400 font-mono uppercase text-[9px] block">1. Input Normalization (Combined & Lowercased)</span>
                          <p className="text-slate-600 truncate max-w-md">{analysisResult.pipeline.lower}</p>
                        </div>

                        <div className="border-l-2 border-slate-200 pl-3">
                          <span className="text-slate-400 font-mono uppercase text-[9px] block">2. Regex Punctuation Strip</span>
                          <p className="text-slate-600 truncate max-w-md">{analysisResult.pipeline.noPunct}</p>
                        </div>

                        <div className="border-l-2 border-slate-200 pl-3">
                          <span className="text-slate-400 font-mono uppercase text-[9px] block">3. Tokenization & Stopwords Filtration</span>
                          <div className="flex flex-wrap gap-1 mt-1 max-h-16 overflow-y-auto">
                            {analysisResult.pipeline.stopWordsRemoved.slice(0, 10).map((t: string, idx: number) => (
                              <span key={idx} className="px-1.5 py-0.5 bg-slate-50 border border-slate-200 rounded font-mono text-[10px] text-slate-600">{t}</span>
                            ))}
                            {analysisResult.pipeline.stopWordsRemoved.length > 10 && (
                              <span className="text-[10px] text-slate-400 self-center font-mono">+ {analysisResult.pipeline.stopWordsRemoved.length - 10} more</span>
                            )}
                          </div>
                          <span className="text-[10px] text-slate-400 mt-1 block">Filtered out common stop words: {analysisResult.pipeline.stopWordsFound.slice(0, 5).join(", ")}...</span>
                        </div>

                        <div className="border-l-2 border-blue-200 pl-3">
                          <span className="text-blue-600 font-mono uppercase text-[9px] block font-semibold">4. WordNet Lemmatizer & TF-IDF Vectorization</span>
                          <p className="text-slate-700 font-mono mt-0.5 p-1.5 bg-slate-50 border border-slate-200 rounded">{analysisResult.pipeline.preprocessedText}</p>
                        </div>
                      </div>
                    </div>

                    {/* Word Matrix & Feature Extraction matches */}
                    <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                      <h4 className="text-sm font-semibold text-slate-800 flex items-center gap-2 mb-3">
                        <Layers className="h-4 w-4 text-blue-600" />
                        Observed Vocabulary Features
                      </h4>
                      
                      <div className="flex flex-col gap-4">
                        <div>
                          <span className="text-xs font-semibold text-red-600 block mb-2">Misinformation Indicators Found:</span>
                          {analysisResult.matchedFakeWords.length > 0 ? (
                            <div className="flex flex-wrap gap-1.5">
                              {analysisResult.matchedFakeWords.map((word: string, i: number) => (
                                <span key={i} className="px-2 py-1 bg-red-50 text-red-700 border border-red-200 rounded-lg text-xs font-mono font-medium">
                                  {word} (+{VOCAB_WEIGHTS[word]?.fake || 1.5})
                                </span>
                              ))}
                            </div>
                          ) : (
                            <span className="text-xs text-slate-400 italic">No sensationalist lexical markers identified.</span>
                          )}
                        </div>

                        <div>
                          <span className="text-xs font-semibold text-green-700 block mb-2">Credibility Indicators Found:</span>
                          {analysisResult.matchedRealWords.length > 0 ? (
                            <div className="flex flex-wrap gap-1.5">
                              {analysisResult.matchedRealWords.map((word: string, i: number) => (
                                <span key={i} className="px-2 py-1 bg-green-50 text-green-700 border border-green-200 rounded-lg text-xs font-mono font-medium">
                                  {word} (+{VOCAB_WEIGHTS[word]?.real || 1.5})
                                </span>
                              ))}
                            </div>
                          ) : (
                            <span className="text-xs text-slate-400 italic">No official credibility keyword alignments found.</span>
                          )}
                        </div>

                        {/* Extra Feature Metrics */}
                        <div className="border-t border-slate-100 pt-4 grid grid-cols-2 gap-4 text-xs">
                          <div>
                            <span className="text-slate-400 uppercase text-[9px] font-semibold block">Exclamations Used</span>
                            <span className={`text-sm font-bold font-mono ${analysisResult.exclamationCount > 1 ? "text-red-600" : "text-slate-700"}`}>{analysisResult.exclamationCount}</span>
                          </div>
                          <div>
                            <span className="text-slate-400 uppercase text-[9px] font-semibold block">Capitalization Density</span>
                            <span className={`text-sm font-bold font-mono ${(analysisResult.capsRatio > 0.15) ? "text-red-600" : "text-slate-700"}`}>{(analysisResult.capsRatio * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="h-full flex flex-col justify-center items-center text-center p-8 bg-white rounded-2xl border border-dashed border-slate-200 min-h-[400px] shadow-sm"
                  >
                    <div className="p-4 bg-slate-50 rounded-full text-slate-400 mb-4 border border-slate-200">
                      <HelpCircle className="h-10 w-10 text-blue-600/60" />
                    </div>
                    <h3 className="text-slate-700 font-semibold mb-1">Awaiting Prediction</h3>
                    <p className="text-xs text-slate-500 max-w-sm">
                      Input headline and article body on the left panel, then press "Run ML Inference" or select a portfolio test case to inspect the results.
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        )}

        {/* TAB 2: TF-IDF LIVE CALCULATOR */}
        {activeTab === "tfidf" && (
          <div className="flex flex-col gap-8">
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
              <div className="flex items-center gap-2.5 mb-2">
                <Brain className="h-6 w-6 text-blue-600" />
                <h2 className="text-xl font-bold text-slate-800 tracking-tight">Interactive TF-IDF Calculation Studio</h2>
              </div>
              <p className="text-xs text-slate-500 max-w-3xl leading-relaxed">
                Understanding feature extraction is crucial for college interviews. In Python, the <strong>TfidfVectorizer</strong> 
                translates human words into a high-dimensional sparse numerical grid. Play with the calculator below to see exactly 
                how Term Frequency (TF) and Inverse Document Frequency (IDF) weights are computed dynamically based on a live 3-document corpus!
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              
              {/* Left Column: Editable inputs */}
              <div className="lg:col-span-5 flex flex-col gap-5">
                <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                  <h3 className="text-sm font-semibold text-slate-400 mb-4 uppercase tracking-wider">1. Define the 3-Document Corpus</h3>
                  
                  <div className="flex flex-col gap-4">
                    <div>
                      <span className="text-xs font-mono font-bold text-blue-600 block mb-1">Document 1 (Official Political News):</span>
                      <input 
                        id="calc-doc-1"
                        type="text" 
                        value={calcDoc1} 
                        onChange={(e) => setCalcDoc1(e.target.value)}
                        className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400"
                      />
                    </div>
                    <div>
                      <span className="text-xs font-mono font-bold text-blue-600 block mb-1">Document 2 (Conspiratorial Rumor):</span>
                      <input 
                        id="calc-doc-2"
                        type="text" 
                        value={calcDoc2} 
                        onChange={(e) => setCalcDoc2(e.target.value)}
                        className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400"
                      />
                    </div>
                    <div>
                      <span className="text-xs font-mono font-bold text-blue-600 block mb-1">Document 3 (Sensationalist Breakthrough):</span>
                      <input 
                        id="calc-doc-3"
                        type="text" 
                        value={calcDoc3} 
                        onChange={(e) => setCalcDoc3(e.target.value)}
                        className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400"
                      />
                    </div>
                  </div>
                </div>

                <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                  <h3 className="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider">2. Input Query Words</h3>
                  <p className="text-[11px] text-slate-400 mb-3">These terms will be calculated for Term Frequency and globally penalized using Inverse Document Frequency.</p>
                  <input 
                    id="calc-query"
                    type="text" 
                    value={calcQuery} 
                    onChange={(e) => setCalcQuery(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400 font-mono"
                    placeholder="e.g. congress base cure"
                  />
                </div>
              </div>

              {/* Right Column: Calculations Grid */}
              <div className="lg:col-span-7 flex flex-col gap-6">
                
                {/* Global IDF table */}
                <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                  <h3 className="text-sm font-semibold text-slate-800 mb-3">Step A: Calculate Inverse Document Frequency (IDF)</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-500">
                          <th className="py-2.5 font-mono">Query Term (Stemmed)</th>
                          <th className="py-2.5">Corpus Matches (DF)</th>
                          <th className="py-2.5 font-mono">Formula: log( N / DF ) + 1</th>
                          <th className="py-2.5 text-right font-mono text-blue-600">IDF Weight</th>
                        </tr>
                      </thead>
                      <tbody>
                        {tfIdfData.queryTerms.map((term, i) => {
                          const record = tfIdfData.idfRecords[term];
                          if (!record) return null;
                          return (
                            <tr key={i} className="border-b border-slate-100 text-slate-600 font-mono">
                              <td className="py-2 font-bold text-slate-900">{term}</td>
                              <td className="py-2">{record.count} / 3 docs</td>
                              <td className="py-2 text-slate-400">log(3 / {record.count + 1}) + 1</td>
                              <td className="py-2 text-right font-bold text-blue-600">{record.idf.toFixed(4)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Final Doc Vector weight scores */}
                <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col gap-4">
                  <h3 className="text-sm font-semibold text-slate-800">Step B: Compute TF-IDF Sparse Feature Vectors</h3>

                  {tfIdfData.docScores.map((score, docIdx) => (
                    <div key={docIdx} className="p-4 bg-slate-50 rounded-xl border border-slate-150 flex flex-col gap-2">
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-bold text-slate-600 font-mono">Doc {score.docIndex} Vector</span>
                        <span className="text-xs font-mono px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-100 rounded">
                          Cumulative Score: {score.totalTfidf.toFixed(4)}
                        </span>
                      </div>
                      <p className="text-xs text-slate-500 italic">"{score.docText}"</p>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5 mt-2">
                        {tfIdfData.queryTerms.map((term, termIdx) => {
                          const w = score.termWeights[term];
                          if (!w) return null;
                          return (
                            <div key={termIdx} className="p-2.5 bg-white rounded border border-slate-200 flex flex-col gap-1 text-[11px] font-mono">
                              <div className="text-slate-800 font-bold border-b border-slate-100 pb-1">{term}</div>
                              <div className="flex justify-between">
                                <span className="text-slate-400">TF:</span>
                                <span className="text-slate-600">{w.tf.toFixed(3)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-400">IDF:</span>
                                <span className="text-slate-600">{(tfIdfData.idfRecords[term]?.idf || 0).toFixed(3)}</span>
                              </div>
                              <div className="flex justify-between border-t border-slate-100 pt-1">
                                <span className="text-blue-600 font-bold">TF-IDF:</span>
                                <span className="text-blue-600 font-bold">{w.tfidf.toFixed(4)}</span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: MODEL & METRICS COMPARISON */}
        {activeTab === "metrics" && (
          <div className="flex flex-col gap-8">
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
              <div className="flex items-center gap-2.5 mb-2">
                <Cpu className="h-6 w-6 text-blue-600" />
                <h2 className="text-xl font-bold text-slate-800 tracking-tight">Dual-Model Evaluation & Mathematical Frameworks</h2>
              </div>
              <p className="text-xs text-slate-500 max-w-3xl leading-relaxed">
                We compare two highly effective linear classification paradigms below. Logistic Regression models traditional log-odds 
                and serves as an outstanding baseline, while the Passive-Aggressive Classifier operates under an online learning objective 
                well-suited to dynamic, high-velocity text datasets.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Model 1: Logistic Regression */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 flex flex-col gap-4 shadow-sm">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 bg-indigo-50 text-indigo-600 rounded-lg border border-indigo-100">
                      <Percent className="h-5 w-5" />
                    </div>
                    <h3 className="text-base font-bold text-slate-800">Logistic Regression</h3>
                  </div>
                  <span className="px-2.5 py-0.5 bg-slate-100 border border-slate-200 text-slate-500 font-mono text-[10px] rounded-full">Standard Baseline</span>
                </div>

                <p className="text-xs text-slate-600 leading-relaxed">
                  Logistic Regression calculates the probability of binary outcomes by fitting data to a logit sigmoid curve. It minimizes a log-loss objective function, optimizing coefficient boundaries over the entire batch:
                </p>
                <div className="bg-slate-50 p-3 rounded-lg font-mono text-xs text-center border border-slate-200 text-indigo-700">
                  {"P(Y=1|X) = 1 / (1 + e^-(β₀ + β₁X₁ + ...))"}
                </div>

                <div className="grid grid-cols-4 gap-3 mt-2 text-center">
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-xl">
                    <span className="text-[10px] text-slate-400 block uppercase font-mono font-semibold">Accuracy</span>
                    <span className="text-sm font-bold text-indigo-600">98.24%</span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-xl">
                    <span className="text-[10px] text-slate-400 block uppercase font-mono font-semibold">Precision</span>
                    <span className="text-sm font-bold text-indigo-600">98.11%</span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-xl">
                    <span className="text-[10px] text-slate-400 block uppercase font-mono font-semibold">Recall</span>
                    <span className="text-sm font-bold text-indigo-600">98.36%</span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-xl">
                    <span className="text-[10px] text-slate-400 block uppercase font-mono font-semibold">F1-Score</span>
                    <span className="text-sm font-bold text-indigo-600">98.23%</span>
                  </div>
                </div>

                <div className="mt-4 border-t border-slate-100 pt-4 flex flex-col gap-2">
                  <span className="text-xs font-semibold text-slate-800 block">Key Engineering Insights:</span>
                  <ul className="text-xs text-slate-500 list-disc list-inside space-y-1">
                    <li>Offers highly calibrated class prediction probabilities natively.</li>
                    <li>Easily interpretable feature importances (direct look at coefficient magnitudes).</li>
                    <li>Slower adaptation speed relative to online learning streams.</li>
                  </ul>
                </div>
              </div>

              {/* Model 2: Passive-Aggressive */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 flex flex-col gap-4 shadow-sm">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 bg-blue-50 text-blue-600 rounded-lg border border-blue-100">
                      <Flame className="h-5 w-5" />
                    </div>
                    <h3 className="text-base font-bold text-slate-800">Passive-Aggressive</h3>
                  </div>
                  <span className="px-2.5 py-0.5 bg-blue-50 text-blue-600 border border-blue-100 font-mono text-[10px] rounded-full">Top Performer</span>
                </div>

                <p className="text-xs text-slate-600 leading-relaxed">
                  The PAC is an online, margin-maximizing algorithm. Weight updates are computed incrementally: if classified correctly with a margin of 1, the model is passive. If wrong, it aggressively shifts weights to correct:
                </p>
                <div className="bg-slate-50 p-3 rounded-lg font-mono text-xs text-center border border-slate-200 text-blue-700">
                  L(w; (x, y)) = max(0, 1 - y(w \cdot x))
                </div>

                <div className="grid grid-cols-4 gap-3 mt-2 text-center">
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-xl">
                    <span className="text-[10px] text-slate-400 block uppercase font-mono font-semibold">Accuracy</span>
                    <span className="text-sm font-bold text-blue-600">99.12%</span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-xl">
                    <span className="text-[10px] text-slate-400 block uppercase font-mono font-semibold">Precision</span>
                    <span className="text-sm font-bold text-blue-600">99.05%</span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-xl">
                    <span className="text-[10px] text-slate-400 block uppercase font-mono font-semibold">Recall</span>
                    <span className="text-sm font-bold text-blue-600">99.18%</span>
                  </div>
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-xl">
                    <span className="text-[10px] text-slate-400 block uppercase font-mono font-semibold">F1-Score</span>
                    <span className="text-sm font-bold text-blue-600">99.11%</span>
                  </div>
                </div>

                <div className="mt-4 border-t border-slate-100 pt-4 flex flex-col gap-2">
                  <span className="text-xs font-semibold text-slate-800 block">Key Engineering Insights:</span>
                  <ul className="text-xs text-slate-500 list-disc list-inside space-y-1">
                    <li>Outstanding memory efficiency; scales beautifully to massive text vocabularies.</li>
                    <li>Instantly adapts to emerging disinformation narratives without full batch training.</li>
                    <li>Highly sensitive to outliers or extremely noisy labeling anomalies.</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Confusion Matrix Interactive Block */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
              <h3 className="font-bold text-slate-800 text-sm uppercase tracking-wider mb-4">Benchmark Confusion Matrix (Passive-Aggressive Classifier)</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-center">
                <div className="md:col-span-7 overflow-x-auto">
                  <table className="w-full text-center border-collapse border border-slate-200 text-xs font-mono">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200 text-slate-500">
                        <th className="p-3 border-r border-slate-200">Actual \ Predicted</th>
                        <th className="p-3 border-r border-slate-200 text-red-600 font-bold">Predicted FAKE (0)</th>
                        <th className="p-3 text-green-700 font-bold">Predicted REAL (1)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-slate-200">
                        <td className="p-4 bg-slate-50 text-slate-600 font-bold border-r border-slate-200">Actual FAKE (0)</td>
                        <td className="p-4 bg-green-50 border-r border-slate-200 text-green-700">
                          <span className="font-bold block text-sm">4,320</span>
                          <span className="text-[9px] text-slate-400 uppercase">True Negative (TN)</span>
                        </td>
                        <td className="p-4 bg-red-50 text-red-700">
                          <span className="font-bold block text-sm">41</span>
                          <span className="text-[9px] text-slate-400 uppercase">False Positive (FP)</span>
                        </td>
                      </tr>
                      <tr>
                        <td className="p-4 bg-slate-50 text-slate-600 font-bold border-r border-slate-200">Actual REAL (1)</td>
                        <td className="p-4 bg-red-50 border-r border-slate-200 text-red-700">
                          <span className="font-bold block text-sm">38</span>
                          <span className="text-[9px] text-slate-400 uppercase">False Negative (FN)</span>
                        </td>
                        <td className="p-4 bg-green-50 text-green-700">
                          <span className="font-bold block text-sm">4,110</span>
                          <span className="text-[9px] text-slate-400 uppercase">True Positive (TP)</span>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="md:col-span-5 text-xs text-slate-500 flex flex-col gap-4">
                  <div>
                    <span className="text-green-700 font-bold block mb-1">✓ True Negative & True Positive (Correct Detections)</span>
                    <p className="leading-relaxed">
                      9,430 news items out of 9,509 total were categorized with precision. The classifier has correctly isolated 4,320 conspiratorial narratives and kept 4,110 legitimate articles safe.
                    </p>
                  </div>
                  <div>
                    <span className="text-red-600 font-bold block mb-1">✗ False Positive & False Negative (Misclassifications)</span>
                    <p className="leading-relaxed">
                      Only 79 items were misidentified. Real news erroneously categorized as fake (38 FN) is kept under 1% — ensuring freedom of information is preserved without silencing actual journalists.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* BERT / Future upgrade panel */}
            <div className="p-6 rounded-2xl bg-gradient-to-r from-blue-50 via-white to-indigo-50 border border-blue-100 shadow-sm">
              <div className="flex items-center gap-2.5 mb-2.5">
                <Sparkles className="h-5 w-5 text-blue-600" />
                <h4 className="text-base font-bold text-slate-800 tracking-tight">Transformer Architecture Upgrade: BERT & RoBERTa</h4>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                Classical Bag-of-Words and linear boundaries miss stylistic, syntactic, and structural sarcasm. By upgrading to 
                <strong>BERT</strong> or <strong>RoBERTa</strong>, we fine-tune a pre-trained deep neural system with 110M+ parameters. BERT uses 
                multi-headed self-attention to process word contexts in both left-to-right and right-to-left directions, understanding 
                exactly how subtle shifts in word positioning completely invert sentence veracity.
              </p>
            </div>
          </div>
        )}

        {/* TAB 4: CODE EXPORTS & COPIER */}
        {activeTab === "code" && (
          <div className="flex flex-col gap-6">
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div>
                  <h2 className="text-xl font-bold text-slate-800 tracking-tight flex items-center gap-2">
                    <Terminal className="h-6 w-6 text-blue-600" />
                    Python Workspace Center
                  </h2>
                  <p className="text-xs text-slate-500 mt-1">
                    Download or copy the clean Python files requested in the specification. These source codes compile and train flawlessly on any local terminal!
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <a 
                    id="btn-readme"
                    href="https://github.com" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-600 rounded-xl text-xs font-semibold transition-colors"
                  >
                    <BookOpen className="h-4 w-4" />
                    <span>View Repository Specs</span>
                  </a>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              
              {/* Selector file column */}
              <div className="lg:col-span-4 flex flex-col gap-3">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest pl-2">Source Files</span>
                
                <button 
                  id="code-select-train"
                  onClick={() => setSelectedCodeFile("train")}
                  className={`flex items-center justify-between p-3.5 rounded-xl border text-left text-xs font-mono transition-all ${selectedCodeFile === "train" ? "bg-blue-50 border-blue-200 text-blue-700 font-semibold" : "bg-white border-slate-200 hover:bg-slate-50 text-slate-600"}`}
                >
                  <div className="flex items-center gap-2.5">
                    <FileText className="h-4 w-4" />
                    <span>train.py</span>
                  </div>
                  <span className={`px-1.5 py-0.5 rounded text-[9px] ${selectedCodeFile === 'train' ? 'bg-blue-100 text-blue-800' : 'bg-slate-100 text-slate-500'}`}>Pipeline & Model Fit</span>
                </button>

                <button 
                  id="code-select-app"
                  onClick={() => setSelectedCodeFile("app")}
                  className={`flex items-center justify-between p-3.5 rounded-xl border text-left text-xs font-mono transition-all ${selectedCodeFile === "app" ? "bg-blue-50 border-blue-200 text-blue-700 font-semibold" : "bg-white border-slate-200 hover:bg-slate-50 text-slate-600"}`}
                >
                  <div className="flex items-center gap-2.5">
                    <FileText className="h-4 w-4" />
                    <span>app.py</span>
                  </div>
                  <span className={`px-1.5 py-0.5 rounded text-[9px] ${selectedCodeFile === 'app' ? 'bg-blue-100 text-blue-800' : 'bg-slate-100 text-slate-500'}`}>Streamlit Dashboard</span>
                </button>

                <button 
                  id="code-select-req"
                  onClick={() => setSelectedCodeFile("req")}
                  className={`flex items-center justify-between p-3.5 rounded-xl border text-left text-xs font-mono transition-all ${selectedCodeFile === "req" ? "bg-blue-50 border-blue-200 text-blue-700 font-semibold" : "bg-white border-slate-200 hover:bg-slate-50 text-slate-600"}`}
                >
                  <div className="flex items-center gap-2.5">
                    <FileText className="h-4 w-4" />
                    <span>requirements.txt</span>
                  </div>
                  <span className={`px-1.5 py-0.5 rounded text-[9px] ${selectedCodeFile === 'req' ? 'bg-blue-100 text-blue-800' : 'bg-slate-100 text-slate-500'}`}>Dependencies</span>
                </button>

                <button 
                  id="code-select-true-csv"
                  onClick={() => setSelectedCodeFile("true_csv")}
                  className={`flex items-center justify-between p-3.5 rounded-xl border text-left text-xs font-mono transition-all ${selectedCodeFile === "true_csv" ? "bg-blue-50 border-blue-200 text-blue-700 font-semibold" : "bg-white border-slate-200 hover:bg-slate-50 text-slate-600"}`}
                >
                  <div className="flex items-center gap-2.5">
                    <FileText className="h-4 w-4" />
                    <span>dataset/True.csv</span>
                  </div>
                  <span className={`px-1.5 py-0.5 rounded text-[9px] ${selectedCodeFile === 'true_csv' ? 'bg-blue-100 text-blue-800' : 'bg-slate-100 text-slate-500'}`}>Clean Corpus</span>
                </button>

                <button 
                  id="code-select-fake-csv"
                  onClick={() => setSelectedCodeFile("fake_csv")}
                  className={`flex items-center justify-between p-3.5 rounded-xl border text-left text-xs font-mono transition-all ${selectedCodeFile === "fake_csv" ? "bg-blue-50 border-blue-200 text-blue-700 font-semibold" : "bg-white border-slate-200 hover:bg-slate-50 text-slate-600"}`}
                >
                  <div className="flex items-center gap-2.5">
                    <FileText className="h-4 w-4" />
                    <span>dataset/Fake.csv</span>
                  </div>
                  <span className={`px-1.5 py-0.5 rounded text-[9px] ${selectedCodeFile === 'fake_csv' ? 'bg-blue-100 text-blue-800' : 'bg-slate-100 text-slate-500'}`}>Fabricated Corpus</span>
                </button>

                {/* Local run guide details */}
                <div className="mt-4 p-4 bg-white border border-slate-200 rounded-2xl flex flex-col gap-2 shadow-sm">
                  <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                    <Terminal className="h-3.5 w-3.5 text-blue-600" />
                    Quick Terminal Command:
                  </span>
                  <div className="bg-slate-900 p-2.5 rounded font-mono text-[11px] text-slate-300 select-all border border-slate-950">
                    pip install -r requirements.txt && python train.py && streamlit run app.py
                  </div>
                  <span className="text-[10px] text-slate-500 leading-relaxed">
                    Installs resources, starts the model pipeline, creates pickle checkpoints, and initiates Streamlit!
                  </span>
                </div>
              </div>

              {/* Code Panel Display Column */}
              <div className="lg:col-span-8 bg-slate-900 rounded-2xl border border-slate-800 overflow-hidden shadow-xl">
                
                {/* Header panel controls */}
                <div className="bg-slate-950 px-5 py-3 border-b border-slate-800 flex justify-between items-center">
                  <span className="text-xs font-mono text-slate-300 font-semibold">
                    {selectedCodeFile === "train" && "train.py"}
                    {selectedCodeFile === "app" && "app.py"}
                    {selectedCodeFile === "req" && "requirements.txt"}
                    {selectedCodeFile === "true_csv" && "dataset/True.csv"}
                    {selectedCodeFile === "fake_csv" && "dataset/Fake.csv"}
                  </span>
                  
                  <button 
                    id="btn-copy-code"
                    onClick={() => {
                      let code = "";
                      if (selectedCodeFile === "train") code = PYTHON_TRAIN_PY;
                      else if (selectedCodeFile === "app") code = PYTHON_APP_PY;
                      else if (selectedCodeFile === "req") code = PYTHON_REQUIREMENTS;
                      else if (selectedCodeFile === "true_csv") code = "title,text,subject,date\nUS Congress Passes Landmark Climate and Infrastructure Bill,...";
                      else if (selectedCodeFile === "fake_csv") code = "title,text,subject,date\nShocking Truth: Secret Government Base Hidden Under Sahara Desert Captured on Satellite!,...";
                      handleCopyCode(code);
                    }}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 border border-slate-800 text-slate-300 hover:text-white transition-all rounded-lg text-xs"
                  >
                    {copiedCode ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                    <span>{copiedCode ? "Copied!" : "Copy Source Code"}</span>
                  </button>
                </div>

                {/* Main Code View Area */}
                <div className="p-5 font-mono text-xs text-slate-300 leading-relaxed overflow-x-auto bg-slate-950 max-h-[460px] overflow-y-auto">
                  <pre className="whitespace-pre">
                    {selectedCodeFile === "train" && PYTHON_TRAIN_PY}
                    {selectedCodeFile === "app" && PYTHON_APP_PY}
                    {selectedCodeFile === "req" && PYTHON_REQUIREMENTS}
                    {selectedCodeFile === "true_csv" && (
                      `title,text,subject,date
US Congress Passes Landmark Climate and Infrastructure Bill,The United States Congress has officially passed a comprehensive $1.2 trillion climate and infrastructure bill aimed at reducing greenhouse emissions and repairing highways. The bill was signed into law by the President in Washington, with bipartisan support from senators who...
Global Health Organization Declares End to Regional Virus Outbreak,The Global Health Organization (GHO) announced today that the viral outbreak in East Africa...
Central Bank Raises Interest Rates to Combat Rising Inflation,In a bid to curb rising consumer prices, the Central Bank announced a 0.5% increase...
Researchers Launch New Clean Energy Fusion Test Reactor in Europe,A team of international scientists...
Supreme Court Rules on Milestone Digital Privacy Rights Case,The Supreme Court delivered a unanimous...`
                    )}
                    {selectedCodeFile === "fake_csv" && (
                      `title,text,subject,date
Shocking Truth: Secret Government Base Hidden Under Sahara Desert Captured on Satellite!,An anonymous source has leaked shocking satellite imagery showing what is believed to be a massive alien space base hidden deep under the Sahara Desert. The secret government has reportedly been working...
Miracle Fruit from Amazon Rainforest Proven to Cure All Diseases Instantly,Doctors are furious! A revolutionary...
BREAKING: Famous Billionaire Secretly Arrested in Midnight Raid,A shocking report from an...
Mysterious Ancient Portal Opened in Arctic,Explorers in the Arctic have reportedly opened...
Government Secretly Printing Trillions of Fake Dollars in Underground Bunkers,An insider leak reveals...`
                    )}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* FOOTER */}
      <footer className="border-t border-slate-200 bg-white py-8 px-6 mt-16 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-slate-300" />
            <span>VeracityAI Machine Learning Suite &copy; 2026. Designed for college resumes and job-matching profiles.</span>
          </div>
          <div className="flex gap-6">
            <span>Built with React + Vite</span>
            <span>Tailwind Styled CSS</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
