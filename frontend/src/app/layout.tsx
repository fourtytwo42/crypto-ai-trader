import "./globals.css";

import type { Metadata } from "next";
import { Fraunces, Space_Grotesk } from "next/font/google";

const sans = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap"
});

const display = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap"
});

export const metadata: Metadata = {
  title: "Crypto AI Trader | Forecast Report",
  description: "AI-assisted crypto market forecast dashboard with live predictions."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${sans.variable} ${display.variable}`}>
        <div className="relative min-h-screen">
          <div className="noise-overlay" />
          {children}
        </div>
      </body>
    </html>
  );
}
