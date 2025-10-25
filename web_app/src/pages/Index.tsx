import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Download, Type, AlignLeft, AlignCenter } from "lucide-react";
import { toast } from "sonner";

type HandwritingStyle = {
  id: number;
  label: string;
};

type TextAlignment = "left" | "center";

type AlignmentOption = {
  value: TextAlignment;
  label: string;
  icon: typeof AlignLeft;
};

const alignmentOptions: AlignmentOption[] = [
  { value: "left", label: "Left", icon: AlignLeft },
  { value: "center", label: "Center", icon: AlignCenter },
];

const MIN_PREVIEW_DELAY_MS = 400;

const Index = () => {
  const [text, setText] = useState("Hello World!\nWelcome to Handwriting Studio");
  const [styles, setStyles] = useState<HandwritingStyle[]>([]);
  const [stylesError, setStylesError] = useState<string | null>(null);
  const [selectedStyleId, setSelectedStyleId] = useState<number | null>(null);
  const [alignment, setAlignment] = useState<TextAlignment>("center");
  const [isDownloading, setIsDownloading] = useState(false);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [previewSvg, setPreviewSvg] = useState<string>("");
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStyles = async () => {
      try {
        const response = await fetch("/api/styles");
        if (!response.ok) {
          throw new Error("Unable to load handwriting styles");
        }

        const data: { styles: HandwritingStyle[] } = await response.json();
        setStyles(data.styles);
        setSelectedStyleId((current) => {
          if (current !== null && data.styles.some((style) => style.id === current)) {
            return current;
          }
          return data.styles.length > 0 ? data.styles[0].id : null;
        });
      } catch (error) {
        console.error(error);
        setStylesError(error instanceof Error ? error.message : "Unable to load handwriting styles");
        setStyles([]);
        setSelectedStyleId(null);
      }
    };

    fetchStyles();
  }, []);

  useEffect(() => {
    if (!text.trim()) {
      setPreviewSvg("");
      setPreviewError(null);
      setIsPreviewLoading(false);
      return;
    }

    const controller = new AbortController();
    const timeout = setTimeout(async () => {
      setIsPreviewLoading(true);
      setPreviewError(null);

      try {
        const response = await fetch("/api/preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            style: selectedStyleId,
            alignment,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const errorBody = await response.json().catch(() => null);
          const message = errorBody?.error ?? "Unable to render preview";
          throw new Error(message);
        }

        const data: { svg: string } = await response.json();
        setPreviewSvg(data.svg);
        setPreviewError(null);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        console.error(error);
        setPreviewSvg("");
        setPreviewError(error instanceof Error ? error.message : "Unable to render preview");
      } finally {
        if (!controller.signal.aborted) {
          setIsPreviewLoading(false);
        }
      }
    }, MIN_PREVIEW_DELAY_MS);

    return () => {
      controller.abort();
      clearTimeout(timeout);
    };
  }, [text, selectedStyleId, alignment]);

  const previewContent = useMemo(() => {
    if (!text.trim()) {
      return <p className="text-muted-foreground">Enter text to see the preview.</p>;
    }

    if (previewError) {
      return <p className="text-destructive font-medium">{previewError}</p>;
    }

    if (isPreviewLoading) {
      return <p className="text-muted-foreground italic">Rendering preview…</p>;
    }

    if (!previewSvg) {
      return <p className="text-muted-foreground">Preview will appear here shortly.</p>;
    }

    return (
      <div
        className="w-full overflow-auto"
        dangerouslySetInnerHTML={{ __html: previewSvg }}
      />
    );
  }, [isPreviewLoading, previewError, previewSvg, text]);

  const downloadSVG = async () => {
    if (!text.trim()) {
      toast.error("Please enter some text before downloading.");
      return;
    }

    setIsDownloading(true);
    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          style: selectedStyleId,
          alignment,
        }),
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => null);
        const message = errorBody?.error ?? "Unable to download SVG";
        throw new Error(message);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `handwriting-${Date.now()}.svg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      toast.success("SVG downloaded successfully!");
    } catch (error) {
      console.error(error);
      toast.error(error instanceof Error ? error.message : "Unable to download SVG");
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/30">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8 text-center">
          <div className="inline-flex items-center gap-2 mb-2">
            <Type className="w-8 h-8 text-primary" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Handwriting Studio
            </h1>
          </div>
          <p className="text-muted-foreground">Transform your text into beautiful handwriting</p>
        </header>

        <div className="grid lg:grid-cols-2 gap-6 max-w-7xl mx-auto">
          <Card className="p-6 space-y-6 shadow-lg">
            <div>
              <label className="block text-sm font-semibold mb-2 text-foreground">
                Your Text
              </label>
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter your text here..."
                className="min-h-[200px] resize-none text-base"
              />
              <p className="text-xs text-muted-foreground mt-2">
                Tip: keep each line under 75 characters for best results.
              </p>
            </div>

            <div>
              <label className="block text-sm font-semibold mb-2 text-foreground">
                Handwriting Style
              </label>
              <Select
                value={selectedStyleId !== null ? String(selectedStyleId) : undefined}
                onValueChange={(value) => setSelectedStyleId(Number(value))}
                disabled={styles.length === 0}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder={stylesError ?? "Select a style"} />
                </SelectTrigger>
                <SelectContent className="bg-popover z-50">
                  {styles.map((style) => (
                    <SelectItem key={style.id} value={String(style.id)} className="cursor-pointer">
                      {style.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {stylesError && (
                <p className="text-destructive text-sm mt-2">{stylesError}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-semibold mb-2 text-foreground">
                Text Alignment
              </label>
              <Select
                value={alignment}
                onValueChange={(value) => setAlignment(value as TextAlignment)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select alignment" />
                </SelectTrigger>
                <SelectContent className="bg-popover z-50">
                  {alignmentOptions.map((option) => {
                    const IconComponent = option.icon;
                    return (
                      <SelectItem key={option.value} value={option.value} className="cursor-pointer">
                        <div className="flex items-center gap-2">
                          <IconComponent className="w-4 h-4" />
                          <span>{option.label}</span>
                        </div>
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>

            <Button
              onClick={downloadSVG}
              className="w-full bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity"
              size="lg"
              disabled={isDownloading || !text.trim()}
            >
              <Download className="w-4 h-4 mr-2" />
              {isDownloading ? "Preparing SVG…" : "Download SVG"}
            </Button>
          </Card>

          <Card className="p-6 flex items-center justify-center shadow-lg bg-white dark:bg-card">
            <div className="w-full h-full min-h-[400px] flex items-center justify-center">
              {previewContent}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Index;
