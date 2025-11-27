import React, { FormEvent, useState } from "react";

const API_BASE_URL = "http://localhost:8000";

type GeneratedImage = {
  data_url: string;
  seed: number | null;
};

type GenerateResponse = {
  images: GeneratedImage[];
  model: string;
  device: string;
  height: number;
  width: number;
  num_inference_steps: number;
  max_sequence_length: number;
};

function App() {
  const [prompt, setPrompt] = useState(
    "A cat holding a sign that says hello world, cinematic lighting"
  );
  const [width, setWidth] = useState(768);
  const [height, setHeight] = useState(768);
  const [steps, setSteps] = useState(4);
  const [maxSeq, setMaxSeq] = useState(256);
  const [numImages, setNumImages] = useState(1);
  const [seed, setSeed] = useState<string>("");
  const [format, setFormat] = useState<"png" | "jpeg">("png");

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateResponse | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    const payload = {
      prompt,
      width: Number(width),
      height: Number(height),
      num_inference_steps: Number(steps),
      max_sequence_length: Number(maxSeq),
      num_images: Number(numImages),
      seed: seed.trim() === "" ? null : Number(seed),
      output_format: format,
    };

    try {
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(
          `Server error (${response.status}): ${
            text || response.statusText || "Unknown error"
          }`
        );
      }

      const data = (await response.json()) as GenerateResponse;
      setResult(data);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Unknown error occurred";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>FLUX.1 [schnell] Text-to-Image</h1>
        <p className="subtitle">
          Fast rectified-flow generation in 1–4 steps, powered by FLUX.1
          [schnell].
        </p>
      </header>

      <main className="app-main">
        <section className="panel panel-form">
          <form onSubmit={handleSubmit}>
            <label className="field">
              <span>Prompt</span>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={4}
                required
              />
            </label>

            <div className="field-group">
              <label className="field">
                <span>Width (px)</span>
                <input
                  type="number"
                  min={256}
                  max={1536}
                  step={16}
                  value={width}
                  onChange={(e) => setWidth(Number(e.target.value))}
                />
              </label>
              <label className="field">
                <span>Height (px)</span>
                <input
                  type="number"
                  min={256}
                  max={1536}
                  step={16}
                  value={height}
                  onChange={(e) => setHeight(Number(e.target.value))}
                />
              </label>
            </div>

            <div className="field-group">
              <label className="field">
                <span>Inference steps (1–4)</span>
                <input
                  type="number"
                  min={1}
                  max={4}
                  value={steps}
                  onChange={(e) => setSteps(Number(e.target.value))}
                />
              </label>
              <label className="field">
                <span>Max sequence length (≤ 256)</span>
                <input
                  type="number"
                  min={16}
                  max={256}
                  value={maxSeq}
                  onChange={(e) => setMaxSeq(Number(e.target.value))}
                />
              </label>
            </div>

            <div className="field-group">
              <label className="field">
                <span># Images</span>
                <input
                  type="number"
                  min={1}
                  max={4}
                  value={numImages}
                  onChange={(e) => setNumImages(Number(e.target.value))}
                />
              </label>
              <label className="field">
                <span>Seed (optional)</span>
                <input
                  type="number"
                  placeholder="Random if empty"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                />
              </label>
            </div>

            <label className="field">
              <span>Output format</span>
              <select
                value={format}
                onChange={(e) => setFormat(e.target.value as "png" | "jpeg")}
              >
                <option value="png">PNG</option>
                <option value="jpeg">JPEG</option>
              </select>
            </label>

            <button className="btn-primary" type="submit" disabled={isLoading}>
              {isLoading ? "Generating…" : "Generate"}
            </button>

            {error && <p className="error">Error: {error}</p>}
          </form>

          <p className="hint">
            FLUX.1 [schnell] is timestep‑distilled and optimized for 1–4 steps
            with guidance scale fixed at 0.
          </p>
        </section>

        <section className="panel panel-results">
          <h2>Results</h2>
          {!result && !isLoading && (
            <p className="placeholder">Submit a prompt to generate images.</p>
          )}

          {isLoading && <p className="placeholder">Generating images…</p>}

          {result && (
            <>
              <div className="meta">
                <span>Model: {result.model}</span>
                <span>Device: {result.device}</span>
                <span>
                  Resolution: {result.width}×{result.height}
                </span>
                <span>Steps: {result.num_inference_steps}</span>
              </div>
              <div className="image-grid">
                {result.images.map((img, idx) => (
                  <figure key={idx} className="image-card">
                    <img src={img.data_url} alt={`Generated ${idx + 1}`} />
                    <figcaption>
                      Image {idx + 1}
                      {img.seed !== null && (
                        <span className="seed-tag">Seed: {img.seed}</span>
                      )}
                      <a
                        href={img.data_url}
                        download={`flux-schnell-${idx + 1}.${format}`}
                      >
                        Download
                      </a>
                    </figcaption>
                  </figure>
                ))}
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
