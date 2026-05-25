import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AutoRadionuclide — MIBG Flagship Run",
  description:
    "Read-only dashboard visualizing a recorded run of the AutoRadionuclide in-silico radioligand discovery engine.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased bg-white text-gray-900 min-h-screen">
        {children}
      </body>
    </html>
  );
}
