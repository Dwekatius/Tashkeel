import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://arabictashkeel.com"),
  title: {
    default: "Sukoon — Arabic Tashkeel Engine | arabictashkeel.com",
    template: "%s | Sukoon Arabic Tashkeel",
  },
  description:
    "Sukoon is a specialist Arabic tashkeel engine with 99.29% DER accuracy. Add Arabic diacritization to any pipeline — faster, cheaper, and more accurate than any LLM. Built for production.",
  keywords: [
    "Arabic tashkeel",
    "Arabic diacritization",
    "tashkeel engine",
    "Arabic vocalization",
    "Arabic NLP",
    "Arabic text processing",
    "harakat",
    "Arabic diacritics",
    "Arabic TTS preprocessing",
    "Arabic machine learning",
    "diacritization model",
    "sukoon",
  ],
  authors: [{ name: "Sukoon" }],
  creator: "Sukoon",
  publisher: "Sukoon",
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true },
  },
  alternates: {
    canonical: "https://arabictashkeel.com",
  },
  openGraph: {
    type: "website",
    url: "https://arabictashkeel.com",
    siteName: "Sukoon Arabic Tashkeel",
    title: "Sukoon — The Arabic Tashkeel Engine Built for Production",
    description:
      "99.29% DER accuracy. Sub-2ms latency. Orders of magnitude cheaper than any LLM. Sukoon is the specialist Arabic diacritization engine for production pipelines.",
    locale: "en_US",
    images: [
      {
        url: "/assets/logo_2_modified.png",
        width: 1200,
        height: 630,
        alt: "Sukoon Arabic Tashkeel Engine",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Sukoon — Arabic Tashkeel Engine",
    description:
      "99.29% DER accuracy. Sub-2ms latency. The specialist Arabic diacritization engine built for production.",
    images: ["/assets/logo_2_modified.png"],
  },
  icons: {
    icon: "/assets/sukoon-logo.png",
    apple: "/assets/sukoon-logo.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "SoftwareApplication",
              name: "Sukoon",
              applicationCategory: "DeveloperApplication",
              operatingSystem: "Linux, Windows, macOS",
              url: "https://arabictashkeel.com",
              description:
                "Sukoon is a specialist Arabic tashkeel (diacritization) engine. It achieves 99.29% DER accuracy with sub-2ms latency, designed for production pipelines.",
              offers: {
                "@type": "Offer",
                priceCurrency: "USD",
                price: "0",
                availability: "https://schema.org/InStock",
              },
              aggregateRating: {
                "@type": "AggregateRating",
                ratingValue: "5",
                ratingCount: "1",
              },
            }),
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
