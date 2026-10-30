import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import App from "./App";

function jsonResponse(payload: unknown) {
  return Promise.resolve({
    ok: true,
    json: () => Promise.resolve(payload),
  });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("research dashboard", () => {
  it("navigates to the simulator page", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/health")) return jsonResponse({ status: "healthy" });
      return jsonResponse({
        total_experiments: 0,
        completed: 0,
        running: 0,
        failed: 0,
        best_experiment: null,
        latest_experiment: null,
      });
    }));

    render(<BrowserRouter><App /></BrowserRouter>);
    await screen.findByRole("heading", { name: "Research dashboard" });
    fireEvent.click(screen.getByRole("link", { name: "Simulator" }));

    expect(await screen.findByRole("heading", { name: "Run a single episode" })).toBeTruthy();
  });

  it("submits an experiment to the backend", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/health")) return jsonResponse({ status: "healthy" });
      if (url.endsWith("/api/dashboard")) {
        return jsonResponse({
          total_experiments: 0,
          completed: 0,
          running: 0,
          failed: 0,
          best_experiment: null,
          latest_experiment: null,
        });
      }
      if (url.endsWith("/api/experiments") && init?.method === "POST") {
        return jsonResponse({ id: 4, status: "queued" });
      }
      return jsonResponse({});
    }));

    render(<BrowserRouter><App /></BrowserRouter>);
    fireEvent.click(screen.getByRole("link", { name: "Experiments" }));
    fireEvent.change(screen.getByLabelText("Experiment name"), { target: { value: "UI run" } });
    fireEvent.click(screen.getByRole("button", { name: "Create experiment" }));

    await waitFor(() => expect(screen.getByText(/Experiment #4 queued/)).toBeTruthy());
  });
});
