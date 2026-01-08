import "./globals.css";
import { Space_Grotesk, IBM_Plex_Sans } from "next/font/google";

const display = Space_Grotesk({ subsets: ["latin"], variable: "--font-display" });
const body = IBM_Plex_Sans({ subsets: ["latin"], weight: ["300", "400", "500", "600"], variable: "--font-body" });

export const metadata = {
  title: "Pump.fun Signal Lab",
  description: "Minute-level prediction and insight for pump.fun tokens.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body className="min-h-screen bg-fog text-ink">
        {children}
      </body>
    </html>
  );
}
