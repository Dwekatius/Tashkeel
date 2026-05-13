"use client";

import { useEffect, useRef, useState } from "react";
import {
  Accessibility,
  ArrowRight,
  Check,
  Code2,
  Cpu,
  Database,
  DollarSign,
  Gauge,
  MessageSquareText,
  Mic2,
  Network,
  Package,
  Rocket,
  Server,
  ShieldCheck,
  Smartphone,
  Sparkles,
  Timer,
  Volume2,
  Zap,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const navItems = ["Model", "Accuracy", "Performance", "Message", "Pricing"];

const heroStats = [
  { value: "99.29%", label: "DER quality", detail: "task-aware scoring" },
  { value: "97.00%", label: "WER quality", detail: "clean punctuation input" },
  { value: "9,281", label: "words/sec GPU", detail: "RTX 6000 Ada batch eval" },
  { value: "438.5", label: "words/sec CPU", detail: "Ryzen 5 service run" },
  { value: "~73 MB", label: "checkpoint", detail: "small local footprint" },
];

const useCases = [
  { label: "Mobile", icon: Smartphone, detail: "A tiny checkpoint that can fit practical product constraints." },
  { label: "Edge devices", icon: Cpu, detail: "Useful where a 30B+ local LLM is unrealistic." },
  { label: "TTS", icon: Mic2, detail: "Adds vowels, endings, shadda, sukoon, tanween, and pause behavior before speech." },
  { label: "IoT", icon: Network, detail: "Text-in/text-out behavior for small Arabic interfaces." },
  { label: "Accessibility", icon: Accessibility, detail: "Improves pronunciation support for reading and assistive flows." },
  { label: "Publishing", icon: MessageSquareText, detail: "Preserves the base text while adding the missing vocalization layer." },
];

const architectureSteps = [
  "Undiacritized Arabic",
  "Unicode cleaning",
  "Arabic segmentation",
  "Character encoder",
  "Transformer context",
  "Diacritic classifier",
  "Orthographic post-processing",
  "Diacritized Arabic",
];

function IconFlamingSpeedo({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      {/* Gauge arc */}
      <path d="M4 17a8 8 0 0 1 16 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      {/* Bold needle pointing top-right (max speed) */}
      <line x1="12" y1="17" x2="18" y2="11" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
      {/* Hub dot */}
      <circle cx="12" cy="17" r="1.6" fill="currentColor" />
      {/* Flame — teardrop shape above gauge */}
      <path d="M12 4 C10 6 9 8 9.5 10 C10 11.5 11 12 12 12 C13 12 14 11.5 14.5 10 C15 8 14 6 12 4 Z" fill="currentColor" opacity="0.9" />
      {/* Inner flame highlight */}
      <path d="M12 7 C11.2 8.2 11 9.2 11.5 10 C11.8 10.6 12.2 10.6 12.5 10 C13 9.2 12.8 8.2 12 7 Z" fill="white" opacity="0.35" />
    </svg>
  );
}

function IconStorageStack({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      {/* Top ellipse */}
      <ellipse cx="12" cy="5.5" rx="8" ry="2.5" stroke="currentColor" strokeWidth="1.9" fill="currentColor" fillOpacity="0.25" />
      {/* Side walls */}
      <path d="M4 5.5 V16.5" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" />
      <path d="M20 5.5 V16.5" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" />
      {/* Middle ring */}
      <path d="M4 10 C4 11.4 7.6 12.5 12 12.5 S20 11.4 20 10" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" fill="none" />
      {/* Bottom ellipse */}
      <ellipse cx="12" cy="16.5" rx="8" ry="2.5" stroke="currentColor" strokeWidth="1.9" fill="currentColor" fillOpacity="0.15" />
    </svg>
  );
}

function IconFlamingCPU({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      {/* Flame */}
      <path d="M12 2 C10.5 4 10 5.5 10.5 7 C11 8 11.5 8.5 12 8.5 C12.5 8.5 13 8 13.5 7 C14 5.5 13.5 4 12 2 Z" fill="currentColor" opacity="0.95" />
      <path d="M12 4 C11.4 5.2 11.2 6 11.5 7 C11.7 7.5 12 7.5 12.3 7 C12.7 6.2 12.5 5 12 4 Z" fill="white" opacity="0.35" />
      {/* Chip body */}
      <rect x="6" y="9" width="12" height="11" rx="1.5" stroke="currentColor" strokeWidth="1.9" fill="currentColor" fillOpacity="0.2" />
      {/* Inner core */}
      <rect x="9.5" y="12" width="5" height="5" rx="0.6" fill="currentColor" />
      {/* Left pins */}
      <path d="M6 12 H4.5 M6 14.5 H4.5 M6 17 H4.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      {/* Right pins */}
      <path d="M18 12 H19.5 M18 14.5 H19.5 M18 17 H19.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function IconLightningClock({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      {/* Clock ring */}
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
      {/* Subtle clock hands (thin) */}
      <line x1="12" y1="7" x2="12" y2="12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" opacity="0.55" />
      <line x1="12" y1="12" x2="15" y2="14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" opacity="0.55" />
      {/* Bold lightning bolt */}
      <path d="M13.5 6.5 L8.5 13.5 L12 13.5 L10.5 17.5 L15.5 10.5 L12 10.5 Z" fill="currentColor" />
    </svg>
  );
}

const performanceCards = [
  {
    icon: Gauge,
    value: "200×",
    label: "faster than SOTA LLM",
    copy: "9,281 words/sec GPU vs 46.5 words/sec from GPT-5.5 — specialist inference vs autoregressive generation.",
    grad: "linear-gradient(135deg, #0ea5e9, #2563eb)",
    tint: "rgba(14,165,233,0.07)",
    borderClr: "rgba(14,165,233,0.22)",
  },
  {
    icon: Package,
    value: "~820×",
    label: "smaller than Gemma 4 30B FP16",
    copy: "73 MB checkpoint vs ~60 GB for a dense 30B FP16 model. Fits on hardware where LLMs simply cannot.",
    grad: "linear-gradient(135deg, #f97316, #c2410c)",
    tint: "rgba(249,115,22,0.07)",
    borderClr: "rgba(249,115,22,0.22)",
  },
  {
    icon: Zap,
    value: "~10×",
    label: "CPU-only beats a 30B on GPU",
    copy: "438.5 words/sec on a plain CPU outpaces a 30B model on an RTX 4090 and surpasses every SOTA cloud API.",
    grad: "linear-gradient(135deg, #8b5cf6, #6d28d9)",
    tint: "rgba(139,92,246,0.07)",
    borderClr: "rgba(139,92,246,0.22)",
  },
  {
    icon: Timer,
    value: "< 2ms",
    label: "per sentence on GPU",
    copy: "Near-zero pipeline overhead. Drop Sukoon into any Arabic pipeline — a 15-word sentence clears in under 2ms.",
    grad: "linear-gradient(135deg, #10b981, #047857)",
    tint: "rgba(16,185,129,0.07)",
    borderClr: "rgba(16,185,129,0.22)",
  },
];

const costRows = [
  ["Sukoon", "TBD", "~1.8 min GPU / ~38.0 min CPU", "Specialist API target + measured throughput"],
  ["GPT-5.5", "≈ $52.50", "~6.1 hours", "$7.50 input + $45.00 output; 69.8 output tok/s"],
  ["Claude Opus 4.7", "≈ $45.00", "~10.0 hours", "$7.50 input + $37.50 output; 42.2 output tok/s"],
];

const llmComparison = [
  ["Native Arabic tashkeel specialization", "Purpose-built", "Prompted behavior"],
  ["Text preservation", "Designed to preserve text", "Can rewrite or mutate"],
  ["Latency profile", "Small deterministic pass", "Network + generation latency"],
  ["TTS pipeline fit", "Excellent preprocessor", "Expensive and prompt-sensitive"],
];

const runSteps = [
  { icon: Code2, title: "Clone", body: "git clone https://github.com/Dwekatious/Tashkeel.git" },
  { icon: Database, title: "Start", body: "docker compose up --build" },
  { icon: Rocket, title: "Open", body: "http://127.0.0.1:8765/" },
  { icon: ShieldCheck, title: "Ship", body: "Use clean text-in/text-out output in your Arabic workflow." },
];

const pricingPlans = [
  {
    name: "Pro",
    price: "TBD",
    note: "Placeholder plan for small teams testing Arabic content workflows.",
    features: ["Local web UI", "Batch playground", "Basic support"],
  },
  {
    name: "Ultra",
    price: "TBD",
    note: "Placeholder plan for heavier preprocessing, education, and publishing pipelines.",
    features: ["Higher throughput", "Team workspace", "Priority runtime"],
  },
  {
    name: "API",
    price: "TBD",
    note: "Placeholder API tier for product integration, TTS, and embedded Arabic systems.",
    features: ["REST API", "Usage-based billing", "Private deployment path"],
  },
];

export default function Home() {
  useEffect(() => {
    const elements = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.18, rootMargin: "0px 0px -8% 0px" },
    );

    elements.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, []);

  return (
    <main className="min-h-screen overflow-x-hidden bg-[#fbfcff] text-slate-950">
      <nav className="site-nav fixed inset-x-0 top-0 z-40">
        <div className="nav-shell mx-auto flex items-center justify-between gap-8">
          <div className="nav-links hidden items-center gap-1 lg:flex">
            {navItems.map((item) => (
              <a key={item} href={`#${item.toLowerCase()}`} className="nav-link">
                <span>{item}</span>
              </a>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <a href="#" className="nav-action hidden sm:inline-flex">
              <span>Log in</span>
            </a>
          </div>
        </div>
      </nav>

      <section id="model" className="hero-section relative isolate overflow-hidden">
        <div className="home-bg-layer" />
        <div className="hero-logo-watermark" aria-hidden="true">
          <img src="/assets/sukoon-logo.png" alt="" />
        </div>
        <div className="hero-shell mx-auto">
          <div className="hero-welcome relative">
            <div className="absolute inset-0 hero-mesh" />
            <div className="hero-split-line" aria-hidden="true" />
            <div className="hero-motion-stripes" aria-hidden="true" />
            <div className="hero-content-grid relative grid p-14 xl:p-20">
              <div className="hero-left-stack">
                <div className="hero-logo-panel">
                  <img src="/assets/logo_2_modified.png" alt="Sukoon" className="hero-logo-2" />
                </div>

                <div className="hero-copy-panel">
                  <Badge variant="warm" className="mb-5">
                    <Sparkles className="size-3.5" />
                    Specialist Arabic diacritization
                  </Badge>
                  <h1 className="text-6xl font-black leading-[0.96] text-slate-950 xl:text-7xl">
                    Arabic tashkeel,<br />done right.
                  </h1>
                  <p className="mt-7 max-w-3xl text-xl leading-9 text-slate-600">
                    Sukoon is a purpose-built Arabic diacritization engine — 99.29% accurate, sub-2ms fast,
                    and a fraction of the cost of any LLM. Built for pipelines that cannot afford to get it wrong.
                  </p>
                </div>
              </div>

              <div className="hero-right-stack">
                <MiniDemo />

                <Button variant="accent" className="hero-primary-cta"
                  onClick={() => { window.location.href = "/app.html?v=" + Date.now(); }}>
                  <span className="cta-copy">
                    <span>Try Sukoon</span>
                    <small>open the live demo</small>
                  </span>
                  <ArrowRight className="cta-arrow" />
                </Button>
              </div>

              <div className="hero-dials-row">
        <HeroDials />
      </div>
            </div>
          </div>
        </div>
      </section>

      <section id="accuracy" className="page-section bg-white">
        <SectionIntro
          eyebrow="Accuracy"
          title="Task-aware scoring for the marks Sukoon is actually asked to predict."
          copy="The filtered benchmark reports tashkeel quality on clean, punctuation-correct text, ignoring hamza spelling and pause-form final-haraka differences that the runtime intentionally normalizes."
        />

        <div className="mt-12 grid grid-cols-[1.08fr_0.92fr] gap-8">
          <Card data-reveal className="reveal-card overflow-hidden">
            <CardHeader>
              <CardTitle>DER and WER quality</CardTitle>
              <CardDescription>Task-aware benchmark scores on clean, punctuation-correct Arabic text.</CardDescription>
            </CardHeader>
            <CardContent>
              <AnimatedAccuracyVisual />
            </CardContent>
          </Card>

          <Card data-reveal data-delay={1} className="reveal-card accuracy-story-card">
            <CardHeader>
              <CardDescription>Specialist accuracy</CardDescription>
              <CardTitle>Built for the task that general LLMs have to be talked into doing.</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 text-base leading-8 text-slate-600">
              <p>
                Arabic tashkeel is a precision layer, not an open-ended writing task. Large 30B-class local models and even
                SOTA hosted LLMs can reason about Arabic, but when the job is to vocalize existing text, Sukoon is the better
                fit: it is trained to preserve the original words, punctuation, and sentence order while adding the missing
                harakat. On the filtered benchmark it reaches 99.29% DER quality and 97.00% WER quality across 113,955
                evaluated words.
              </p>
              <p>
                That kind of predictable vocalization can push Arabic TTS into a new era: clearer pronunciation, better
                endings, shadda, tanween, pause handling, and more natural speech without a huge generative model in the
                loop. It also matters for education, from children learning to read to non-native learners studying Arabic,
                because the text stays the same while pronunciation support becomes visible. LLM prompts can hallucinate,
                paraphrase, normalize spelling, drop punctuation, or slowly drift when asked to process a full book, paper,
                or long document; Sukoon keeps the contract narrow and repeatable.
              </p>
            </CardContent>
          </Card>
        </div>

        <div className="mt-8 grid grid-cols-[0.9fr_1.1fr] gap-8">
          <Card data-reveal className="reveal-card repeatability-card">
            <CardHeader>
              <CardTitle>Repeatability is the product feature.</CardTitle>
              <CardDescription>Same input, same contract, no surprise rewrites.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5 text-base leading-8 text-slate-600">
              <p>
                LLMs are generative systems. Even the best ones can hallucinate, paraphrase, change words, shift sentence
                structure, drop punctuation, or answer differently when you ask the same question again. That variability is
                useful for brainstorming, but it is an architectural mismatch for tashkeel, where the source text must remain
                the source text.
              </p>
              <p>
                Sukoon is built around a preservation-first contract: keep the input words and sentence order, add the
                vocalization layer, and make the run repeatable. That matters when processing a book, lesson, paper, script,
                or production TTS corpus, because one changed word can change meaning and one drifting paragraph can break the
                trust of the whole pipeline.
              </p>
              <div className="grid grid-cols-3 gap-3 text-sm font-semibold text-slate-700">
                <span className="rounded-lg bg-white/80 p-3">Text preserved</span>
                <span className="rounded-lg bg-white/80 p-3">Stable reruns</span>
                <span className="rounded-lg bg-white/80 p-3">Long-form safe</span>
              </div>
            </CardContent>
          </Card>

          <Card data-reveal className="reveal-card">
            <CardHeader>
              <CardTitle>Designed with clear scope boundaries.</CardTitle>
              <CardDescription>Sukoon is a tashkeel specialist, not a general Arabic correction engine.</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-3 gap-4 text-sm leading-7 text-slate-600">
              <div className="rounded-lg bg-orange-50 p-4">
                <strong className="block text-slate-950">Tashkeel, not hamza correction</strong>
                Sukoon adds vocalization marks. It is not designed to fix hamza spelling or rewrite the base Arabic text.
              </div>
              <div className="rounded-lg bg-sky-50 p-4">
                <strong className="block text-slate-950">Punctuation matters</strong>
                Sukoon works best with punctuated text because punctuation changes meaning, pause, and flow. Without it,
                the model is forced toward guessing intent.
              </div>
              <div className="rounded-lg bg-rose-50 p-4">
                <strong className="block text-slate-950">Tanween is in scope</strong>
                Sukoon handles tanween, but exact fathatan placement, especially around alif, has orthographic disagreement
                across sources and styles.
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      <section id="performance" className="page-section bg-gradient-to-br from-cyan-50 via-white to-orange-50">
        <SectionIntro
          eyebrow="Performance and cost"
          title="A tiny encoder pass instead of expensive autoregressive generation."
          copy="Sukoon is measured in words per second, not slow token-by-token generation. It is positioned as a specialist utility for high-volume Arabic preprocessing."
        />

        <div className="mt-12 grid grid-cols-4 gap-5">
          {performanceCards.map(({ icon: Icon, value, label, copy, grad, tint, borderClr }, index) => (
            <Card key={label} data-reveal data-delay={index + 1} className="reveal-card metric-card perf-card"
              style={{ background: `radial-gradient(circle at 18% 0%, ${tint}, transparent 55%), white`, borderColor: borderClr }}>
              <CardHeader>
                <div className="mb-3 flex size-12 items-center justify-center rounded-xl text-white" style={{ background: grad }}>
                  <Icon className="size-8" />
                </div>
                <CardDescription>{label}</CardDescription>
                <CardTitle className="perf-card-value text-5xl">{value}</CardTitle>
              </CardHeader>
              <CardContent className="text-sm leading-7 text-slate-600">{copy}</CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-8 grid grid-cols-[1.02fr_0.98fr] gap-8">
          <Card data-reveal className="reveal-card">
            <CardHeader>
              <CardTitle>A specialist, not a generalist</CardTitle>
              <CardDescription>Why purpose-built beats prompted every time.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="philosophy-body">
                <p>
                  Sukoon does not compete with large language models — it renders them irrelevant for this task.
                  Rather than prompting a generative system and hoping it diacritizes faithfully, Sukoon is a
                  custom encoder model built from the ground up with a single purpose: outperform every other
                  model on Arabic diacritization. That focus translates directly into state-of-the-art accuracy —
                  99.29% DER and 97.00% WER — numbers no general-purpose LLM delivers at this cost, speed, or scale.
                </p>
                <p>
                  Because it is a deterministic pipeline rather than a generative model, Sukoon integrates into any
                  existing workflow without the weight that comes with an LLM. No GPU cluster required, no
                  multi-second inference rounds, no unpredictable outputs. With inference at{" "}
                  <span className="philosophy-stat">&lt; 2 ms per sentence</span>, it slots into a production
                  system as quietly as a logging call — barely measurable overhead, immediate results.
                </p>
                <p>
                  Cost is where the gap becomes a landslide. Running Sukoon is{" "}
                  <span className="philosophy-stat">orders of magnitude cheaper</span> than routing text through a
                  cloud LLM API. This is what makes it genuinely different: not just a better model, but a different
                  philosophy. AI as a real specialized tool that solves one problem as close to perfectly as possible,
                  rather than a general system approximating many things adequately. That is the direction
                  high-quality production AI is heading.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card data-reveal className="reveal-card overflow-hidden flex flex-col">
            <CardHeader>
              <CardTitle>Sukoon vs LLM alternatives</CardTitle>
              <CardDescription>Head-to-head across the dimensions that matter in production.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-1 flex-col">
              <FitMapVisual />
            </CardContent>
          </Card>
        </div>

        <Card data-reveal className="reveal-card mt-8">
          <CardHeader>
            <CardTitle>Pipeline trace</CardTitle>
            <CardDescription>
              Follow a real sentence through every stage of Sukoon's character-level encoder.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <PipelineTrace />
          </CardContent>
        </Card>
      </section>

      <section id="message" className="om-section">
        <div className="om-glow om-glow-1" />
        <div className="om-glow om-glow-2" />
        <div className="om-inner">

          <span className="om-badge">Our Message</span>

          <h2 className="om-title">AI is a tool<span className="om-dash"> —</span></h2>
          <p className="om-subtitle">arguably the most powerful one we&apos;ve ever built.</p>

          <div className="om-timeline">
            <div className="om-tl-track" />
            {([
              { label:"Printing Press", year:"1440" },
              { label:"Electricity",    year:"1879" },
              { label:"Internet",       year:"1991" },
              { label:"AI",             year:"Now",  active:true },
            ] as {label:string;year:string;active?:boolean}[]).map(item=>(
              <div key={item.label} className={`om-tl-node${item.active?" om-tl-node--active":""}`}>
                <div className="om-tl-dot"/>
                <span className="om-tl-label">{item.label}</span>
                <span className="om-tl-year">{item.year}</span>
              </div>
            ))}
          </div>

          <div className="om-body">
            <p>We believe in creating things that are honest about what they are and genuinely useful where they are deployed. Our goal is to build technology designed to make people&apos;s lives better — <em>not just to impress investors or fill out a pitch deck.</em></p>
            <p>We don&apos;t see AI as a replacement for human beings; we see it as an <strong>extension of human ingenuity.</strong> Every great leap in civilization — the printing press, electricity, the internet — was powered by tools that expanded what we could do. AI is simply the next step. When used right, it has the potential to improve our quality of life in ways that actually matter: in medicine, education, culture, and the preservation of language and heritage. That potential is real, and it deserves to be taken seriously.</p>
            <p className="om-callout">That&apos;s exactly why we&apos;re so skeptical of how AI is often treated today.</p>
            <p>It&apos;s not a marketing gimmick. It&apos;s not a magic lever to inflate a stock price. It shouldn&apos;t be shoehorned into products where it adds zero value, or tossed around as a buzzword just to sound modern. AI is a technology with real consequences, and consequence demands honesty about where it actually helps and where it doesn&apos;t. <strong>Hype isn&apos;t a strategy; it&apos;s a waste of a pivotal moment.</strong></p>
            <p>We want to create AI that solves real problems — tools that integrate quietly into the way people already work, rather than demanding to replace them entirely. We focus on making things that are measurably better at a specific task than anything else out there.</p>
          </div>

          <div className="om-closing">
            <p className="om-closing-lead">Ultimately, we want to put capability directly into human hands.</p>
            <p className="om-closing-statement">
              Not AI that threatens —<br/>
              <span className="om-closing-accent">AI that empowers.</span>
            </p>
          </div>

        </div>
      </section>

      <section id="pricing" className="page-section bg-gradient-to-br from-rose-50 via-white to-sky-50">
        <SectionIntro
          eyebrow="Pricing and contact"
          title="Dummy plans for the future specialist tashkeel tier."
          copy="Sukoon pricing is still TBD. These plans are intentionally placeholder packaging around the idea that tashkeel should be priced like a specialized utility, not frontier reasoning."
        />

        <div className="mt-12 grid grid-cols-3 gap-6">
          {pricingPlans.map((plan, index) => (
            <Card key={plan.name} data-reveal data-delay={index + 1} className="reveal-card pricing-card">
              <CardHeader>
                <CardDescription>{plan.name}</CardDescription>
                <CardTitle className="text-5xl">{plan.price}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="mb-6 leading-7 text-slate-600">{plan.note}</p>
                <div className="space-y-3">
                  {plan.features.map((feature) => (
                    <div key={feature} className="flex items-center gap-3 text-sm font-semibold text-slate-700">
                      <Check className="size-4 text-emerald-600" />
                      {feature}
                    </div>
                  ))}
                </div>
              </CardContent>
              <CardFooter>
                <Button variant={plan.name === "Ultra" ? "accent" : "outline"} className="w-full">
                  Placeholder CTA
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>

        <div data-reveal className="reveal-card mt-12 rounded-lg border border-slate-200 bg-white p-10">
          <div className="grid grid-cols-[1fr_auto] items-center gap-8">
            <div>
              <Badge variant="mint">
                <DollarSign className="size-3.5" />
                Commercial target
              </Badge>
              <h2 className="mt-5 text-4xl font-black">Make Arabic tashkeel cheap enough to use everywhere.</h2>
              <p className="mt-4 max-w-3xl leading-8 text-slate-600">
                The point is not to compete with LLMs at everything. It is to make one high-volume Arabic operation fast,
                controllable, and inexpensive enough for product teams to use by default.
              </p>
            </div>
            <Button variant="accent" size="lg">
              Contact us
              <ArrowRight />
            </Button>
          </div>
        </div>
      </section>
    </main>
  );
}

function SectionIntro({ eyebrow, title, copy }: { eyebrow: string; title: string; copy: string }) {
  return (
    <div data-reveal className="reveal-card mx-auto max-w-5xl text-center">
      <Badge variant="rose">{eyebrow}</Badge>
      <h2 className="mt-5 text-5xl font-black leading-tight text-slate-950 xl:text-6xl">{title}</h2>
      <p className="mx-auto mt-6 max-w-3xl text-lg leading-8 text-slate-600">{copy}</p>
    </div>
  );
}

function HeroWelcomeVisual() {
  return (
    <div className="hero-showcase" aria-label="Animated Sukoon welcome visual">
      <div className="hero-showcase-grid" />
      <div className="hero-orbit hero-orbit-one" />
      <div className="hero-orbit hero-orbit-two" />
      <div className="hero-orbit hero-orbit-three" />
      <div className="hero-floating-chip hero-chip-top">
        <Zap className="size-4 text-orange-500" />
        9,281 words/sec
      </div>
      <div className="hero-floating-chip hero-chip-left">
        <Package className="size-4 text-sky-500" />
        ~73 MB checkpoint
      </div>
      <div className="hero-floating-chip hero-chip-right">
        <Volume2 className="size-4 text-rose-500" />
        TTS ready
      </div>
      <div className="hero-sample-card" dir="rtl" lang="ar">
        <span>نص عربي</span>
        <strong>كَتَبَ الطَّالِبُ الدَّرْسَ ثُمَّ قَرَأَ صَفْحَةً مِنَ الْكِتَابِ.</strong>
      </div>
    </div>
  );
}

const ACCURACY_STATS = [
  {
    label: "DER",
    pct: 99.29,
    error: 0.71,
    title: "Per-character diacritic accuracy",
    detail: "5,370 char errors across 495,602",
    tint: "sky" as const,
  },
  {
    label: "WER",
    pct: 97.00,
    error: 3.00,
    title: "Per-word full correctness",
    detail: "3,943 word errors across 113,955",
    tint: "rose" as const,
  },
];

function AnimatedAccuracyVisual() {
  const valueRefs = useRef<(HTMLSpanElement | null)[]>([]);

  useEffect(() => {
    const duration = 1500;
    const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

    const targets = ACCURACY_STATS.map(s => s.pct);
    let raf: number;
    let cancelled = false;

    const tStart = setTimeout(() => {
      const start = performance.now();
      function tick(now: number) {
        if (cancelled) return;
        const p = easeOut(Math.min((now - start) / duration, 1));
        valueRefs.current.forEach((el, i) => {
          if (!el) return;
          el.textContent = (targets[i] * p).toFixed(2);
        });
        if (p < 1) raf = requestAnimationFrame(tick);
      }
      raf = requestAnimationFrame(tick);
    }, 250);

    return () => {
      cancelled = true;
      clearTimeout(tStart);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <div className="accuracy-visual">
      <div className="accuracy-header">
        <h3>Tashkeel quality</h3>
        <p>Filtered benchmark · 16,506 aligned rows · 113,955 words evaluated</p>
      </div>
      <div className="accuracy-grid">
        {ACCURACY_STATS.map((s, i) => (
          <div key={s.label} className={`accuracy-stat tint-${s.tint}`}>
            <div className="accuracy-stat-top">
              <span className="accuracy-stat-label">{s.label}</span>
              <span className="accuracy-stat-pill">{s.error.toFixed(2)}% error</span>
            </div>
            <p className="accuracy-stat-title">{s.title}</p>
            <div className="accuracy-stat-value">
              <span ref={(el) => { valueRefs.current[i] = el; }}>0.00</span>
              <small>%</small>
            </div>
            <div className="accuracy-stat-track">
              <div
                className="accuracy-stat-fill"
                style={{ width: `${s.pct}%`, animationDelay: `${0.3 + i * 0.18}s` }}
              />
            </div>
            <p className="accuracy-stat-detail">{s.detail}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function FitMapVisual() {
  const metrics = ["1M words", "Model size", "GPU needed", "Text safe", "Task fit"];

  const rows = [
    {
      name: "Cloud LLM API",
      color: "#be123c",
      bg: "rgba(244,63,94,0.07)",
      border: "rgba(244,63,94,0.18)",
      cells: ["~6 hours", "N/A", "Cloud GPU", "Can rewrite", "Prompted"],
      dots: ["#f43f5e", "#94a3b8", "#94a3b8", "#f43f5e", "#f43f5e"],
    },
    {
      name: "Local LLM",
      color: "#c2410c",
      bg: "rgba(249,115,22,0.07)",
      border: "rgba(249,115,22,0.18)",
      cells: ["~4 hours", "~60 GB FP16", "Yes", "Can rewrite", "Prompted"],
      dots: ["#f97316", "#f97316", "#f43f5e", "#f43f5e", "#f43f5e"],
    },
    {
      name: "Sukoon",
      color: "#065f46",
      bg: "rgba(16,185,129,0.1)",
      border: "rgba(16,185,129,0.28)",
      cells: ["~2 min GPU", "73 MB", "No — CPU fine", "Always", "Purpose-built"],
      dots: ["#10b981", "#10b981", "#10b981", "#10b981", "#10b981"],
    },
  ];

  return (
    <div className="fit-table">
      <div className="fit-table-head">
        <div className="fit-table-corner" />
        {metrics.map((m) => (
          <div key={m} className="fit-table-metric">{m}</div>
        ))}
      </div>
      {rows.map((row, ri) => (
        <div
          key={row.name}
          className="fit-table-row"
          style={{ background: row.bg, borderColor: row.border }}
        >
          <div className="fit-table-name" style={{ color: row.color }}>{row.name}</div>
          {row.cells.map((cell, ci) => (
            <div key={ci} className="fit-table-cell">
              <span className="fit-table-dot" style={{ background: row.dots[ci] }} />
              <span className="fit-table-val" style={{ color: ri === 2 ? row.color : "#475569", fontWeight: ri === 2 ? 800 : 600 }}>
                {cell}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function ArchitectureSketch() {
  return (
    <svg viewBox="0 0 980 430" className="h-[430px] w-full" role="img" aria-label="Sukoon architecture sketch">
      <defs>
        <linearGradient id="flowGradient" x1="0%" x2="100%" y1="0%" y2="0%">
          <stop offset="0%" stopColor="#f97316" />
          <stop offset="50%" stopColor="#f43f5e" />
          <stop offset="100%" stopColor="#0ea5e9" />
        </linearGradient>
      </defs>
      <rect width="980" height="430" rx="8" fill="#ffffff" />
      <path d="M95 146 H885 M885 146 V282 H95" fill="none" stroke="#e2e8f0" strokeWidth="10" strokeLinecap="round" />
      <path className="architecture-flow" d="M95 146 H885 M885 146 V282 H95" fill="none" stroke="url(#flowGradient)" strokeWidth="5" strokeLinecap="round" strokeDasharray="18 18" />
      {architectureSteps.map((label, index) => {
        const topRow = index < 4;
        const x = topRow ? 60 + index * 230 : 750 - (index - 4) * 230;
        const y = topRow ? 88 : 258;
        return (
          <g key={label} className="architecture-node" style={{ animationDelay: `${index * 110}ms` }}>
            <rect x={x} y={y} width="170" height="92" rx="8" fill={index === 0 || index === 7 ? "#fff7ed" : "#f8fafc"} stroke={index === 0 || index === 7 ? "#fb923c" : "#cbd5e1"} strokeWidth="2" />
            <circle cx={x + 28} cy={y + 28} r="12" fill={index === 0 || index === 7 ? "#f97316" : "#0ea5e9"} />
            <text x={x + 85} y={y + 56} textAnchor="middle" fill="#0f172a" fontSize="15" fontWeight="900">
              {label}
            </text>
          </g>
        );
      })}
      <text x="60" y="48" fill="#0f172a" fontSize="28" fontWeight="900">
        Character-level encoder pipeline
      </text>
      <text x="60" y="390" fill="#64748b" fontSize="18" fontWeight="700">
        Clean input, segment Arabic safely, classify diacritics, then apply orthographic post-processing.
      </text>
    </svg>
  );
}

// ── Shared pipeline data ────────────────────────────────────────
const GC = ["#f43f5e","#8b5cf6","#10b981"] as const;
const SC = { height: 190, width: "auto", maxWidth: "100%" } as React.CSSProperties;
const CHARS = [
  { c:"أ",g:0 },{ c:"ك",g:0 },{ c:"ل",g:0 },{ c:" ",g:-1 },
  { c:"ا",g:1 },{ c:"ل",g:1 },{ c:"ط",g:1,f:true },{ c:"ف",g:1 },{ c:"ل",g:1 },{ c:" ",g:-1 },
  { c:"ط",g:2 },{ c:"ع",g:2 },{ c:"ا",g:2 },{ c:"م",g:2 },{ c:"ه",g:2 },{ c:".",g:-1 },
] as const;
const VW = 720, BW = 36, BH = 42, BGAP = 6;
const totalBW = CHARS.length * (BW + BGAP) - BGAP;
const BSX = (VW - totalBW) / 2;
const bcx = (i: number) => BSX + i * (BW + BGAP) + BW / 2;

// ── Step visuals ─────────────────────────────────────────────────

function VisualInput() {
  const words = [
    { ar: "أكل",    role: "verb",    c: GC[0] },
    { ar: "الطفل", role: "subject", c: GC[1] },
    { ar: "طعامه.", role: "object",  c: GC[2] },
  ];
  return (
    <svg viewBox="0 0 720 190" style={SC} aria-hidden="true">
      {/* Arabic sentence with per-word colored backgrounds via HTML */}
      <foreignObject x={60} y={18} width={600} height={82}>
        <div
          // @ts-expect-error xmlns required for SVG foreignObject
          xmlns="http://www.w3.org/1999/xhtml"
          style={{ display:"flex", flexDirection:"row", justifyContent:"center",
                   alignItems:"center", height:"82px", gap:"8px", direction:"rtl" }}
        >
          {words.map(w => (
            <span key={w.ar} style={{
              fontFamily:'"Amiri","Noto Naskh Arabic",serif',
              fontSize:"36px", fontWeight:"700", color:"#0f172a",
              background: w.c + "22", borderRadius:"10px",
              border: `1.5px solid ${w.c}44`, padding:"2px 10px",
              lineHeight: "1.3",
            }}>{w.ar}</span>
          ))}
        </div>
      </foreignObject>
      {/* Stats bar */}
      {[["3 words","#0f172a"],["14 chars","#0f172a"],["0 diacritics","#f43f5e"],["1 punct","#0f172a"]].map(([t,c],i)=>(
        <g key={t as string}>
          <rect x={130+i*120} y={118} width={108} height={30} rx="8" fill="rgba(15,23,42,0.04)" stroke="rgba(15,23,42,0.08)" strokeWidth="1"/>
          <text x={184+i*120} y={138} textAnchor="middle" fill={c as string} fontSize="12" fontWeight="800">{t as string}</text>
        </g>
      ))}
      {/* Legend: colored dot + label */}
      {words.map(({ role, c }, i) => (
        <g key={role}>
          <circle cx={232+i*106} cy={171} r="5" fill={c}/>
          <text x={242+i*106} y={175} fill="#475569" fontSize="12" fontWeight="700">{role}</text>
        </g>
      ))}
    </svg>
  );
}

function VisualClean() {
  const codes = ["0623","0643","0644","0020","0627","0644","0637","0641","0644","0020","0637","0639","0627","0645","0647","002E"];
  return (
    <svg viewBox="0 0 720 190" style={SC} aria-hidden="true">
      <text x={360} y={22} textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="800" letterSpacing="0.06em">
        UNICODE CODEPOINTS — ALL CANONICAL NFC
      </text>
      {CHARS.map((ch,i)=>{
        const x = BSX + i*(BW+BGAP); const col = ch.g>=0 ? GC[ch.g as 0|1|2] : "#e2e8f0";
        const isSpace = ch.g === -1;
        return (
          <g key={i}>
            {!isSpace && <rect x={x} y={36} width={BW} height={BH} rx="7" fill="white" stroke={col} strokeWidth="1.5"/>}
            <text x={x+BW/2} y={64} textAnchor="middle" fill={isSpace?"#e2e8f0":col}
              fontSize="16" fontWeight="700" fontFamily='"Amiri","Noto Naskh Arabic",serif'>{ch.c}</text>
            {!isSpace && <text x={x+BW/2} y={95} textAnchor="middle" fill="#94a3b8" fontSize="8" fontWeight="700"
              fontFamily="ui-monospace,monospace">U+{codes[i]}</text>}
          </g>
        );
      })}
      {[["NFC normalised","#0ea5e9"],["No spurious marks","#10b981"],["Base letters locked","#f97316"]].map(([t,c],i)=>(
        <g key={t as string}>
          <rect x={78+i*190} y={118} width={172} height={30} rx="9" fill={(c as string)+"15"} stroke={(c as string)+"40"} strokeWidth="1"/>
          <text x={78+i*190+10} y={137} fill={c as string} fontSize="12" fontWeight="900">✓</text>
          <text x={78+i*190+26} y={137} fill="#334155" fontSize="11" fontWeight="700">{t as string}</text>
        </g>
      ))}
      <text x={360} y={175} textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="700">
        16 positions · 14 Arabic base letters · 2 whitespace · 1 punct
      </text>
    </svg>
  );
}

function VisualSegment() {
  return (
    <svg viewBox="0 0 720 190" style={SC} aria-hidden="true">
      <text x={360} y={18} textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="800" letterSpacing="0.06em">
        CHARACTER TOKENS — 14 ARABIC + 2 SPACES + 1 PUNCT = 17 POSITIONS
      </text>
      {CHARS.map((ch,i)=>{
        const x = BSX + i*(BW+BGAP); const isSpace = ch.c===" ";
        const isPunct = ch.c==="."; const col = ch.g>=0 ? GC[ch.g as 0|1|2] : isPunct?"#94a3b8":"#e2e8f0";
        return (
          <g key={i}>
            <text x={x+BW/2} y={34} textAnchor="middle" fill="#cbd5e1" fontSize="9" fontWeight="700"
              fontFamily="ui-monospace,monospace">{i}</text>
            {!isSpace && <rect x={x} y={40} width={BW} height={BH} rx="8"
              fill={col+"18"} stroke={col} strokeWidth="1.8"/>}
            <text x={x+BW/2} y={67} textAnchor="middle" fill={isSpace?"#e2e8f0":col}
              fontSize="17" fontWeight="700" fontFamily='"Amiri","Noto Naskh Arabic",serif'>{ch.c}</text>
          </g>
        );
      })}
      {/* Word group brackets */}
      {[{from:0,to:2,label:"verb",c:GC[0]},{from:4,to:8,label:"subject",c:GC[1]},{from:10,to:14,label:"object",c:GC[2]}].map(b=>{
        const lx = BSX+b.from*(BW+BGAP); const rx = BSX+b.to*(BW+BGAP)+BW;
        const my = 98;
        return (
          <g key={b.label}>
            <path d={`M${lx} ${my-6} L${lx} ${my} L${rx} ${my} L${rx} ${my-6}`} fill="none" stroke={b.c} strokeWidth="1.5"/>
            <text x={(lx+rx)/2} y={my+14} textAnchor="middle" fill={b.c} fontSize="10" fontWeight="900">{b.label}</text>
          </g>
        );
      })}
      <text x={360} y={138} textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="700">
        Each Arabic letter is an independent encoder input — whitespace and punct are position markers only.
      </text>
    </svg>
  );
}

function VisualAttention() {
  const focalIdx = 6;
  const arcs = [
    {to:0,s:0.55,d:0},{to:2,s:0.50,d:160},{to:4,s:0.85,d:80},
    {to:5,s:1.00,d:0},{to:7,s:0.90,d:40},{to:8,s:0.78,d:120},
    {to:10,s:0.45,d:200},{to:14,s:0.30,d:280},
  ];
  return (
    <svg viewBox="0 0 720 190" style={SC} aria-label="Transformer attention visualization">
      <text x={360} y={14} textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="800" letterSpacing="0.06em">
        ATTENTION FROM FOCAL TOKEN ط (in الطفل) → ALL 17 POSITIONS
      </text>
      {arcs.map(({to,s,d})=>{
        const x1=bcx(focalIdx), x2=bcx(to), mx=(x1+x2)/2;
        const span=Math.abs(x2-x1), ch=Math.min(72,14+span*0.38), py=105-ch;
        return <path key={to} d={`M${x1} 105 Q${mx} ${py} ${x2} 105`}
          fill="none" stroke="#8b5cf6" strokeWidth={0.8+s*2.2} strokeLinecap="round"
          className="attention-arc" style={{opacity:0.18+s*0.6,animationDelay:`${d}ms`}}/>;
      })}
      {CHARS.map((ch,i)=>{
        const x=BSX+i*(BW+BGAP); const isFocal=(ch as typeof CHARS[number] & {f?:boolean}).f;
        const isSpace=ch.g===-1; const col=ch.g>=0?GC[ch.g as 0|1|2]:"#cbd5e1";
        return (
          <g key={i}>
            {!isSpace&&<rect x={x} y={105} width={BW} height={BH} rx="8"
              fill={isFocal?"#8b5cf6":"white"} stroke={isFocal?"#8b5cf6":col}
              strokeWidth={isFocal?0:1.8} className={isFocal?"attention-focal":""}/>}
            <text x={x+BW/2} y={132} textAnchor="middle"
              fill={isFocal?"white":isSpace?"#e2e8f0":col} fontSize="16" fontWeight="700"
              fontFamily='"Amiri","Noto Naskh Arabic",serif'>{ch.c}</text>
            {isFocal&&<text x={x+BW/2} y={160} textAnchor="middle" fill="#8b5cf6" fontSize="9" fontWeight="900">focal</text>}
          </g>
        );
      })}
      {([["أكل verb",0],["الطفل subject",1],["طعامه object",2]] as [string,0|1|2][]).map(([l,gi],li)=>(
        <g key={l}><rect x={110+li*180} y={170} width={8} height={8} rx="2" fill={GC[gi]}/>
        <text x={124+li*180} y={178} fill="#64748b" fontSize="10" fontWeight="700">{l}</text></g>
      ))}
    </svg>
  );
}

function VisualClassify() {
  const bars = [
    {label:"فَ  fatha",  pct:96, c:"#f43f5e"},
    {label:"فِ  kasra",  pct:2,  c:"#8b5cf6"},
    {label:"فُ  damma",  pct:1,  c:"#f97316"},
    {label:"◌ْ  sukun",  pct:1,  c:"#94a3b8"},
  ];
  return (
    <svg viewBox="0 0 720 190" style={SC} aria-hidden="true">
      <text x={360} y={46} textAnchor="middle" fontFamily='"Amiri","Noto Naskh Arabic",serif'
        fontSize="30" fontWeight="700" fill="#0f172a" direction="rtl">
        أَكَلَ الطِّفْلُ طَعَامَهُ.
      </text>
      <text x={360} y={64} textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="700">
        raw diacritic predictions — pre post-processing
      </text>

      {/* Softmax chart */}
      <text x={36} y={88} fill="#334155" fontSize="11" fontWeight="900">softmax for أ :</text>
      {bars.map((b,i)=>{
        const bw = Math.min(b.pct * 1.35, 160);
        return (
          <g key={b.label}>
            <text x={36} y={106+i*22} fill="#64748b" fontSize="11" fontWeight="700">{b.label}</text>
            <rect x={160} y={95+i*22} width={bw} height={13} rx="4" fill={b.c} opacity="0.85"/>
            <text x={164+bw} y={106+i*22} fill={b.c} fontSize="10" fontWeight="900">{b.pct}%</text>
          </g>
        );
      })}

      {/* 16 classes legend */}
      <text x={420} y={88} fill="#334155" fontSize="11" fontWeight="900">16 classes per token:</text>
      {["fatha · kasra · damma","sukun · shadda","tanween (4 forms)","shadda combos · no-mark"].map((t,i)=>(
        <text key={t} x={420} y={106+i*22} fill="#64748b" fontSize="11" fontWeight="600">• {t}</text>
      ))}

      <text x={360} y={182} textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="700">
        single forward pass classifies all 14 base letters simultaneously
      </text>
    </svg>
  );
}

function VisualPost() {
  return (
    <svg viewBox="0 0 720 190" style={SC} aria-hidden="true">
      <text x={360} y={15} textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="800" letterSpacing="0.06em">
        PAUSE RULE — DROP FINAL SHORT VOWEL BEFORE  .  !  ?  ؟  ؛
      </text>

      {/* Context sentence (small, top) */}
      <text x={680} y={44} textAnchor="end" fontFamily='"Amiri","Noto Naskh Arabic",serif'
        fontSize="20" fontWeight="600" fill="#94a3b8" direction="rtl">
        أَكَلَ الطِّفْلُ طَعَامَهُ.
      </text>

      {/* Zoomed BEFORE */}
      <rect x={70} y={58} width={200} height={60} rx="12" fill="#f43f5e" fillOpacity="0.08" stroke="#f43f5e" strokeOpacity="0.25" strokeWidth="1.5"/>
      <text x={84} y={75} fill="#f43f5e" fontSize="10" fontWeight="900" letterSpacing="0.06em">BEFORE</text>
      <text x={260} y={101} textAnchor="end" fontFamily='"Amiri","Noto Naskh Arabic",serif'
        fontSize="34" fontWeight="700" fill="#f43f5e">طَعَامَهُ.</text>

      {/* Arrow + rule */}
      <text x={360} y={82} textAnchor="middle" fill="#64748b" fontSize="11" fontWeight="700">next char: '.'</text>
      <path d="M 310 92 L 405 92" stroke="#10b981" strokeWidth="2" strokeLinecap="round" markerEnd="url(#arr)"/>
      <defs>
        <marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
          <path d="M0 0 L6 3 L0 6 Z" fill="#10b981"/>
        </marker>
      </defs>
      <text x={360} y={110} textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="800">rule fires → drop ُ</text>

      {/* Zoomed AFTER */}
      <rect x={450} y={58} width={200} height={60} rx="12" fill="#10b981" fillOpacity="0.08" stroke="#10b981" strokeOpacity="0.25" strokeWidth="1.5"/>
      <text x={464} y={75} fill="#10b981" fontSize="10" fontWeight="900" letterSpacing="0.06em">AFTER ✓</text>
      <text x={640} y={101} textAnchor="end" fontFamily='"Amiri","Noto Naskh Arabic",serif'
        fontSize="34" fontWeight="700" fill="#065f46">طَعَامَه.</text>

      {/* Context sentence after (small, bottom) */}
      <text x={680} y={140} textAnchor="end" fontFamily='"Amiri","Noto Naskh Arabic",serif'
        fontSize="20" fontWeight="600" fill="#065f46" direction="rtl">
        أَكَلَ الطِّفْلُ طَعَامَه.
      </text>

      <text x={360} y={170} textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="700">
        damma (ُ) removed from ه · same rule applies before ! ? ؟ ؛ :
      </text>
    </svg>
  );
}

function VisualOutput() {
  const checks = ["Base text identical","Dense tashkeel","Pause-aware","Pipeline-ready"];
  return (
    <svg viewBox="0 0 720 190" style={SC} aria-hidden="true">
      <defs>
        <linearGradient id="outGrad" x1="0%" x2="100%" y1="0%" y2="0%">
          <stop offset="0%" stopColor="#10b981"/><stop offset="100%" stopColor="#059669"/>
        </linearGradient>
      </defs>

      {/* Output sentence */}
      <text x={360} y={54} textAnchor="middle" fontFamily='"Amiri","Noto Naskh Arabic",serif'
        fontSize="36" fontWeight="700" fill="url(#outGrad)" direction="rtl">
        أَكَلَ الطِّفْلُ طَعَامَه.
      </text>
      <text x={360} y={74} textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="700">
        final output — dense vocalization, base text untouched
      </text>

      {/* Check badges */}
      {checks.map((label,i)=>(
        <g key={label}>
          <rect x={18+i*172} y={88} width={160} height={36} rx="10"
            fill="#10b981" fillOpacity="0.09" stroke="#10b981" strokeOpacity="0.3" strokeWidth="1.2"/>
          <text x={34+i*172} y={111} fill="#10b981" fontSize="14" fontWeight="900">✓</text>
          <text x={52+i*172} y={111} fill="#065f46" fontSize="11" fontWeight="800">{label}</text>
        </g>
      ))}

      {/* IN vs OUT comparison */}
      <text x={36} y={150} fill="#94a3b8" fontSize="11" fontWeight="800">IN</text>
      <text x={66} y={150} fontFamily='"Amiri","Noto Naskh Arabic",serif'
        fontSize="15" fill="#94a3b8" textAnchor="start">أكل الطفل طعامه.</text>

      <text x={290} y={150} fill="#10b981" fontSize="11" fontWeight="800">OUT</text>
      <text x={328} y={150} fontFamily='"Amiri","Noto Naskh Arabic",serif'
        fontSize="15" fill="#065f46" textAnchor="start">أَكَلَ الطِّفْلُ طَعَامَه.</text>

      <text x={580} y={150} fill="#94a3b8" fontSize="11" fontWeight="700">0 chars changed</text>
      <text x={360} y={178} textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="700">
        base letter sequence bit-for-bit identical — only diacritics added
      </text>
    </svg>
  );
}

const PIPELINE_STEPS = [
  { label:"Input", short:"Input", color:"#64748b", note:"The raw sentence enters the pipeline with no diacritics. Sukoon receives plain Arabic text — punctuation is present and the base letter sequence is never touched.", visual:"input" },
  { label:"Unicode normalisation", short:"Normalise", color:"#0ea5e9", note:"Unicode codepoints are standardised to their canonical NFC forms. Any stray diacritic marks or encoding artefacts are stripped. The base letter sequence is now locked and verified — 14 Arabic letters, confirmed.", visual:"clean" },
  { label:"Character segmentation", short:"Segment", color:"#8b5cf6", note:"Every Arabic letter becomes an independent input token. Spaces and punctuation markers are kept in their positions. This sentence yields 17 total positions for the encoder.", visual:"segment" },
  { label:"Character encoder", short:"Encode", color:"#f97316", note:"Each character is mapped to a learned embedding. A 6-layer bidirectional transformer then attends across all 17 positions simultaneously — so ط in الطفل knows it is a subject noun before any diacritic is predicted.", visual:"attention" },
  { label:"Diacritic classifier", short:"Classify", color:"#f43f5e", note:"A per-character softmax head predicts one of 16 diacritic classes per base letter: fatha, kasra, damma, sukun, shadda, all four tanween forms, shadda combinations, or no mark. Every letter is classified in a single forward pass.", visual:"classify" },
  { label:"Post-processing", short:"Post", color:"#10b981", note:"Orthographic rules are applied after the neural prediction. The pause rule fires on ه: the damma (ُ) is dropped because the next character is a full stop. Hamza spelling and tanween attachment position are also normalised here.", visual:"post" },
  { label:"Output", short:"Output", color:"#059669", note:"Dense tashkeel — every letter carries its correct diacritic mark. The base text is bit-for-bit identical to the input. The sentence is now ready for TTS synthesis, publishing, accessibility tools, or any downstream Arabic pipeline.", visual:"output" },
];

function PipelineSVG({ current }: { current: number }) {
  const W = 860, H = 88, r = 22;
  const positions = PIPELINE_STEPS.map((_, i) => ({
    x: r + 4 + i * ((W - (r + 4) * 2) / (PIPELINE_STEPS.length - 1)),
    y: 36,
  }));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
      {positions.slice(0,-1).map((pos,i)=>{
        const next=positions[i+1], done=i<current, active=i===current-1;
        return (
          <g key={i}>
            <line x1={pos.x+r} y1={pos.y} x2={next.x-r} y2={next.y} stroke="#e2e8f0" strokeWidth="3" strokeLinecap="round"/>
            {(done||active)&&<line x1={pos.x+r} y1={pos.y} x2={next.x-r} y2={next.y}
              stroke={PIPELINE_STEPS[i+1].color} strokeWidth="3" strokeLinecap="round"
              strokeDasharray={active?"6 4":undefined} className={active?"ptrace-line-active":""}/>}
          </g>
        );
      })}
      {positions.map((pos,i)=>{
        const s=PIPELINE_STEPS[i], isActive=i===current, isDone=i<current, isFuture=i>current;
        return (
          <g key={i}>
            {isActive&&<circle cx={pos.x} cy={pos.y} r={r+9} fill={s.color} opacity="0.15" className="ptrace-glow"/>}
            <circle cx={pos.x} cy={pos.y} r={r} fill={isFuture?"white":s.color} stroke={isFuture?"#e2e8f0":s.color} strokeWidth="2.5"/>
            {isDone&&<text x={pos.x} y={pos.y+5} textAnchor="middle" fill="white" fontSize="14" fontWeight="900">✓</text>}
            {!isDone&&<text x={pos.x} y={pos.y+5} textAnchor="middle" fill={isFuture?"#94a3b8":"white"} fontSize="13" fontWeight="900">{i+1}</text>}
            <text x={pos.x} y={pos.y+r+16} textAnchor="middle" fill={isActive?s.color:isFuture?"#cbd5e1":"#94a3b8"} fontSize="10" fontWeight={isActive?"900":"700"}>{s.short}</text>
          </g>
        );
      })}
    </svg>
  );
}

function PipelineTrace() {
  const [step, setStep] = useState(0);
  const s = PIPELINE_STEPS[step];
  const total = PIPELINE_STEPS.length;
  const visuals: Record<string, React.ReactNode> = {
    input: <VisualInput key={step}/>,
    clean: <VisualClean key={step}/>,
    segment: <VisualSegment key={step}/>,
    attention: <VisualAttention key={step}/>,
    classify: <VisualClassify key={step}/>,
    post: <VisualPost key={step}/>,
    output: <VisualOutput key={step}/>,
  };
  return (
    <div className="ptrace">
      <PipelineSVG current={step}/>
      <div className="ptrace-stage" style={{ borderColor: s.color+"44", background: s.color+"0d" }}>
        {visuals[s.visual]}
      </div>
      <div className="ptrace-note" style={{ borderLeftColor: s.color }}>
        <div className="ptrace-note-label" style={{ color: s.color }}>Step {step+1} — {s.label}</div>
        <p className="ptrace-note-text">{s.note}</p>
      </div>
      <div className="ptrace-nav">
        <button className="ptrace-btn" onClick={()=>setStep(v=>Math.max(0,v-1))} disabled={step===0}>← Prev</button>
        <div className="ptrace-dots">
          {PIPELINE_STEPS.map((_,i)=>(
            <button key={i} className={`ptrace-dot${i===step?" ptrace-dot-active":i<step?" ptrace-dot-done":""}`}
              style={i===step?{background:s.color,width:28}:i<step?{background:s.color+"80"}:{}}
              onClick={()=>setStep(i)} aria-label={`Step ${i+1}`}/>
          ))}
        </div>
        <button className="ptrace-btn" onClick={()=>setStep(v=>Math.min(total-1,v+1))} disabled={step===total-1}>Next →</button>
      </div>
    </div>
  );
}


const DIAL_R = 44;
const DIAL_C = parseFloat((2 * Math.PI * DIAL_R).toFixed(2));
const DIAL_ARC = parseFloat((DIAL_C * (240 / 360)).toFixed(2));

const HERO_DIAL_DATA = [
  { label: "GPU speed",    value: "9,281", unit: "w/s",    pct: 93,    color: "#0ea5e9" },
  { label: "DER quality",  value: "99.29", unit: "%",      pct: 99.29, color: "#f97316" },
  { label: "WER quality",  value: "97.00", unit: "%",      pct: 97,    color: "#f43f5e" },
  { label: "vs SOTA LLM",  value: "200×",  unit: "faster", pct: 100,   color: "#10b981" },
];

function HeroDials() {
  const arcRefs = useRef<(SVGCircleElement | null)[]>([]);

  useEffect(() => {
    const fills = HERO_DIAL_DATA.map(d => parseFloat(((d.pct / 100) * DIAL_ARC).toFixed(2)));
    const duration = 1500;

    function easeOutCubic(t: number) {
      return 1 - Math.pow(1 - t, 3);
    }

    const timers = fills.map((fill, i) => {
      return setTimeout(() => {
        const start = performance.now();
        function frame(now: number) {
          const p = easeOutCubic(Math.min((now - start) / duration, 1));
          const el = arcRefs.current[i];
          if (el) el.style.strokeDashoffset = String(fill * (1 - p));
          if (p < 1) requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
      }, 400 + i * 160);
    });

    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <div className="hero-dials">
      {HERO_DIAL_DATA.map((d, i) => {
        const fill = parseFloat(((d.pct / 100) * DIAL_ARC).toFixed(2));
        return (
          <div key={d.label} className="hero-dial">
            <svg viewBox="0 0 120 108" className="hero-dial-svg" aria-label={`${d.label}: ${d.value} ${d.unit}`}>
              {/* Glow filter */}
              <defs>
                <filter id={`glow-${i}`} x="-30%" y="-30%" width="160%" height="160%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
              </defs>
              {/* Track arc */}
              <circle
                cx="60" cy="66" r={DIAL_R}
                fill="none"
                stroke="rgba(15,23,42,0.08)"
                strokeWidth="8"
                strokeDasharray={`${DIAL_ARC} ${DIAL_C - DIAL_ARC}`}
                strokeLinecap="round"
                transform="rotate(135, 60, 66)"
              />
              {/* Progress arc — starts hidden, JS fills it in */}
              <circle
                ref={(el) => { arcRefs.current[i] = el; }}
                cx="60" cy="66" r={DIAL_R}
                fill="none"
                stroke={d.color}
                strokeWidth="8"
                strokeDasharray={`${fill} ${DIAL_C}`}
                strokeDashoffset={fill}
                strokeLinecap="round"
                transform="rotate(135, 60, 66)"
                filter={`url(#glow-${i})`}
              />
              {/* Value */}
              <text x="60" y="62" textAnchor="middle" fill="#0f172a" fontSize="18" fontWeight="900" fontFamily="Inter, ui-sans-serif, sans-serif">
                {d.value}
              </text>
              {/* Unit */}
              <text x="60" y="78" textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="700" fontFamily="Inter, ui-sans-serif, sans-serif">
                {d.unit}
              </text>
            </svg>
            <p className="hero-dial-label">{d.label}</p>
          </div>
        );
      })}
    </div>
  );
}

const DEMO_EXAMPLES = [
  {
    input: "كتب الطالب الدرس ثم قرأ صفحة من الكتاب.",
    output: "كَتَبَ الطَّالِبُ الدَّرْسَ ثُمَّ قَرَأَ صَفْحَةً مِنَ الْكِتَابِ.",
  },
  {
    input: "السماء صافية اليوم، والشمس تشرق بقوة.",
    output: "السَّمَاءُ صَافِيَةٌ الْيَوْمَ، وَالشَّمْسُ تُشْرِقُ بِقُوَّةٍ.",
  },
  {
    input: "قال المعلم للطلاب: افتحوا الكتاب في الصفحة الأولى.",
    output: "قَالَ الْمُعَلِّمُ لِلطُّلَّابِ: افْتَحُوا الْكِتَابَ فِي الصَّفْحَةِ الْأُولَى.",
  },
  {
    input: "يجب أن نحافظ على البيئة من أجل الأجيال القادمة.",
    output: "يَجِبُ أَنْ نُحَافِظَ عَلَى الْبِيئَةِ مِنْ أَجْلِ الْأَجْيَالِ الْقَادِمَةِ.",
  },
  {
    input: "الكتاب خير جليس في الزمان، فاقرأ دائما.",
    output: "الْكِتَابُ خَيْرُ جَلِيسٍ فِي الزَّمَانِ، فَاقْرَأْ دَائِمًا.",
  },
  {
    input: "أكل الأطفال الموز والتفاح في الحديقة الكبيرة.",
    output: "أَكَلَ الْأَطْفَالُ الْمَوْزَ وَالتُّفَّاحَ فِي الْحَدِيقَةِ الْكَبِيرَةِ.",
  },
];

function MiniDemo() {
  const inputRef = useRef<HTMLParagraphElement>(null);
  const outputRef = useRef<HTMLParagraphElement>(null);
  const procRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    function after(ms: number, fn: () => void) {
      timer = setTimeout(() => { if (!cancelled) fn(); }, ms);
    }

    function setProc(on: boolean) {
      if (procRef.current) procRef.current.style.opacity = on ? "1" : "0";
    }

    let idx = 0;

    function runExample() {
      const { input, output } = DEMO_EXAMPLES[idx];
      const inp = inputRef.current;
      const out = outputRef.current;
      if (!inp || !out || cancelled) return;

      inp.textContent = "";
      out.textContent = "";
      setProc(false);

      let i = 0;
      inp.dataset.cursor = "true";

      function typeInput() {
        if (cancelled) return;
        inp!.textContent = input.slice(0, i++);
        if (i <= input.length) {
          after(42, typeInput);
        } else {
          inp!.dataset.cursor = "false";
          setProc(true);
          after(1100, () => {
            setProc(false);
            let o = 0;
            function typeOutput() {
              if (cancelled) return;
              out!.textContent = output.slice(0, o++);
              if (o <= output.length) {
                after(34, typeOutput);
              } else {
                after(2800, () => {
                  idx = (idx + 1) % DEMO_EXAMPLES.length;
                  after(300, runExample);
                });
              }
            }
            typeOutput();
          });
        }
      }

      after(500, typeInput);
    }

    runExample();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  return (
    <div className="hero-mini-demo" dir="rtl" lang="ar">
      <div className="mini-chrome" dir="ltr">
        <div className="mini-chrome-dots" aria-hidden="true">
          <span /><span /><span />
        </div>
        <span className="mini-chrome-label">Sukoon Engine</span>
        <div className="mini-chrome-live">
          <span className="mini-demo-dot" />
          <span>Live · GPU</span>
        </div>
      </div>

      <div className="mini-demo-body">
        <div className="mini-zone mini-zone-input">
          <span className="mini-zone-tag" dir="ltr">INPUT</span>
          <p ref={inputRef} data-cursor="false">&nbsp;</p>
        </div>

        <div ref={procRef} className="mini-proc" aria-hidden="true" dir="ltr">
          <div className="mini-proc-bar">
            <div className="mini-proc-beam" />
          </div>
          <span className="mini-proc-label">9,281 w/s · GPU</span>
        </div>

        <div className="mini-zone mini-zone-output">
          <span className="mini-zone-tag" dir="ltr">OUTPUT</span>
          <p ref={outputRef}>&nbsp;</p>
        </div>
      </div>

      <div className="mini-demo-footer" dir="ltr">
        <span>99.29% DER</span>
        <span>97.00% WER</span>
        <span>73 MB</span>
        <span>pause aware</span>
      </div>
    </div>
  );
}