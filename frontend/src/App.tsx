import { FormEvent, useEffect, useState } from "react";
import {
  NavLink,
  Route,
  Routes,
  useNavigate,
  useParams,
} from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, Experiment, PolicyEntry, Step } from "./api";

const navigation = [
  ["Overview", "/"],
  ["Simulator", "/simulator"],
  ["Experiments", "/experiments"],
  ["History & compare", "/history"],
  ["Policy inspection", "/policies"],
  ["Methodology", "/methodology"],
];

function App() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">FN</div>
          <div>
            <strong>Fieldnotes</strong>
            <span>RESEARCH PLATFORM</span>
          </div>
        </div>
        <nav aria-label="Main navigation">
          {navigation.map(([label, path]) => (
            <NavLink key={path} to={path} end={path === "/"}>
              <span className="nav-dot" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-note">
          <span className="status-dot" /> Synthetic environment
          <p>Research use only · Not clinical advice</p>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div><span className="eyebrow">RESEARCH WORKSPACE</span><strong>Fatigue-aware nudging</strong></div>
          <HealthIndicator />
        </header>
        <div className="content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route path="/experiments" element={<ExperimentBuilder />} />
            <Route path="/history" element={<History />} />
            <Route path="/history/:id" element={<ExperimentDetailPage />} />
            <Route path="/policies" element={<PolicyInspector />} />
            <Route path="/methodology" element={<Methodology />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

function HealthIndicator() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  useEffect(() => {
    api.health().then(() => setHealthy(true)).catch(() => setHealthy(false));
  }, []);
  return <div className="api-indicator"><span className={`status-dot ${healthy === false ? "offline" : ""}`} />{healthy === null ? "Checking API…" : healthy ? "API connected" : "API unavailable"}</div>;
}

function PageHeading({ kicker, title, text }: { kicker: string; title: string; text: string }) {
  return <div className="page-heading"><span className="eyebrow">{kicker}</span><h1>{title}</h1><p>{text}</p></div>;
}

function ErrorMessage({ message }: { message: string }) {
  return <div className="notice error" role="alert">{message}</div>;
}

function Dashboard() {
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const navigate = useNavigate();
  useEffect(() => { api.dashboard().then(setStats).catch((e: Error) => setError(e.message)); }, []);
  const best = stats?.best_experiment as Experiment | null | undefined;
  const latest = stats?.latest_experiment as Experiment | null | undefined;
  return <>
    <PageHeading kicker="PLATFORM OVERVIEW" title="Research dashboard" text="Explore how policy state representations interact with habituation in a controlled synthetic environment." />
    {error && <ErrorMessage message={error} />}
    <section className="hero-panel">
      <div><span className="eyebrow">RESEARCH QUESTION</span><h2>Can state augmentation prevent habituation failures?</h2><p>Compare adaptive nudging policies with and without access to the simulator’s habituation state.</p>
        <button className="button primary" onClick={() => navigate("/experiments")}>Configure experiment <span aria-hidden="true">↗</span></button>
      </div>
      <div className="hero-art" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit-core">h</div><i className="orbit-node node-one" /><i className="orbit-node node-two" /><i className="orbit-node node-three" /></div>
    </section>
    <div className="metric-grid">
      <Metric label="Experiments" value={stats?.total_experiments ?? "—"} detail={`${stats?.completed ?? 0} completed`} />
      <Metric label="In progress" value={stats?.running ?? "—"} detail={`${stats?.failed ?? 0} failed`} />
      <Metric label="Best observed return" value={best?.mean_discounted_return?.toFixed(2) ?? "—"} detail={best ? best.algorithm : "No completed runs yet"} />
      <Metric label="Latest run" value={latest?.status ?? "—"} detail={latest?.name ?? "Create your first experiment"} />
    </div>
    <section className="panel split-panel">
      <div><span className="eyebrow">MODEL SNAPSHOT</span><h2>Daily health state + habituation</h2><p>The simulator tracks sleep, activity, stress, and a discrete habituation level. A nudge can improve its target dimension when a response occurs; natural drift may lower health values.</p></div>
      <div className="state-chips"><span>s · sleep</span><span>a · activity</span><span>σ · stress</span><span>h · habituation</span></div>
    </section>
    <p className="fine-print">All metrics above are fetched from the local API. The simulator is synthetic and has no clinical validation.</p>
  </>;
}

function Metric({ label, value, detail }: { label: string; value: unknown; detail: string }) {
  return <div className="metric-card"><span>{label}</span><strong>{String(value)}</strong><small>{detail}</small></div>;
}

function Simulator() {
  const [seed, setSeed] = useState(42);
  const [duration, setDuration] = useState(28);
  const [algorithm, setAlgorithm] = useState("rule_based");
  const [fatigueAware, setFatigueAware] = useState(true);
  const [steps, setSteps] = useState<Step[]>([]);
  const [summary, setSummary] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError("");
    try {
      const result = await api.simulation({
        seed, algorithm, fatigue_aware: fatigueAware,
        config: { episode_length: duration },
        algorithm_config: { n_episodes: 100 },
      });
      setSteps(result.steps); setSummary(result.discounted_return);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  return <>
    <PageHeading kicker="INTERACTIVE SIMULATOR" title="Run a single episode" text="Inspect the sequence of actions, responses, health transitions, reward, and habituation for one reproducible seed." />
    <form className="panel form-grid" onSubmit={submit}>
      <label>Episode duration (days)<input type="number" min="1" max="365" value={duration} onChange={e => setDuration(Number(e.target.value))} /></label>
      <label>Random seed<input type="number" min="0" value={seed} onChange={e => setSeed(Number(e.target.value))} /></label>
      <label>Policy<select value={algorithm} onChange={e => setAlgorithm(e.target.value)}><option value="rule_based">Rule-based baseline</option><option value="dp">Dynamic programming</option><option value="mc">Monte Carlo control</option><option value="q_learning">Q-learning</option><option value="oracle_dp">Oracle DP (reference)</option></select></label>
      <label className="checkbox-label"><input type="checkbox" checked={fatigueAware} onChange={e => setFatigueAware(e.target.checked)} /> Fatigue-aware state (where supported)</label>
      <div className="form-actions"><button className="button primary" disabled={busy}>{busy ? "Simulating…" : "Run episode"}</button>{summary !== null && <strong>Discounted return: {summary.toFixed(3)}</strong>}</div>
      {error && <ErrorMessage message={error} />}
    </form>
    {steps.length > 0 && <EpisodeTimeline steps={steps} />}
  </>;
}

function EpisodeTimeline({ steps }: { steps: Step[] }) {
  const chartData = steps.map(step => ({ day: step.day, reward: step.reward, habituation: step.habituation_after, sleep: step.state_after.sleep, activity: step.state_after.activity, stress: step.state_after.stress }));
  return <section className="panel"><div className="section-title"><div><span className="eyebrow">EPISODE TRACE</span><h2>Health & habituation by day</h2></div><span className="pill">{steps.length} days</span></div>
    <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" stroke="#26344a" /><XAxis dataKey="day" stroke="#8c9bb1" /><YAxis stroke="#8c9bb1" /><Tooltip contentStyle={{ background: "#172338", border: "1px solid #35445c", borderRadius: 8 }} /><Line type="monotone" dataKey="sleep" stroke="#59c7b6" dot={false} /><Line type="monotone" dataKey="activity" stroke="#62a8ff" dot={false} /><Line type="monotone" dataKey="stress" stroke="#e0ad65" dot={false} /><Line type="monotone" dataKey="habituation" stroke="#d580a7" dot={false} /></LineChart></ResponsiveContainer></div>
    <div className="table-scroll"><table><thead><tr><th>Day</th><th>Action</th><th>Response p</th><th>Responded</th><th>Reward</th><th>Habituation</th></tr></thead><tbody>{steps.map(step => <tr key={step.day}><td>{step.day}</td><td>{step.action_name}</td><td>{step.response_probability.toFixed(2)}</td><td>{step.responded ? "Yes" : "No"}</td><td>{step.reward.toFixed(3)}</td><td>{step.habituation_before} → {step.habituation_after}</td></tr>)}</tbody></table></div>
  </section>;
}

function ExperimentBuilder() {
  const [name, setName] = useState("Reference run");
  const [algorithm, setAlgorithm] = useState("rule_based");
  const [fatigue, setFatigue] = useState(false);
  const [episodes, setEpisodes] = useState(100);
  const [replications, setReplications] = useState(2);
  const [seed, setSeed] = useState(42);
  const [trainingEpisodes, setTrainingEpisodes] = useState(1000);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();
  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(""); setMessage("");
    try {
      const created = await api.createExperiment({
        name, algorithm, fatigue_aware: fatigue, base_seed: seed,
        n_eval_episodes: episodes, n_replications: replications,
        n_training_episodes: trainingEpisodes,
      });
      setMessage(`Experiment #${created.id} queued. Its status and results are available in history.`);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  return <>
    <PageHeading kicker="EXPERIMENT DESIGN" title="Configure a research run" text="Training and evaluation settings are sent to the backend and persisted with the experiment record." />
    <form className="panel builder-form" onSubmit={submit}>
      <div className="form-grid">
        <label>Experiment name<input value={name} maxLength={255} required onChange={e => setName(e.target.value)} /></label>
        <label>Algorithm<select value={algorithm} onChange={e => setAlgorithm(e.target.value)}><option value="rule_based">Rule-based baseline</option><option value="dp">Dynamic programming</option><option value="mc">Monte Carlo control</option><option value="q_learning">Q-learning</option><option value="oracle_dp">Oracle DP reference</option></select></label>
        <label>Evaluation episodes per replication<input type="number" min="10" max="5000" value={episodes} onChange={e => setEpisodes(Number(e.target.value))} /></label>
        <label>Independent replications<input type="number" min="1" max="20" value={replications} onChange={e => setReplications(Number(e.target.value))} /></label>
        <label>Base seed<input type="number" min="0" value={seed} onChange={e => setSeed(Number(e.target.value))} /></label>
        <label>Training episodes (MC / Q-learning)<input type="number" min="100" max="500000" value={trainingEpisodes} onChange={e => setTrainingEpisodes(Number(e.target.value))} /></label>
      </div>
      <label className="toggle-row"><input type="checkbox" checked={fatigue} onChange={e => setFatigue(e.target.checked)} /><span><strong>Fatigue-aware learner state</strong><small>Include habituation in the algorithm’s state representation.</small></span></label>
      <div className="notice">Simulator defaults follow the workspace specification: 28-day horizon, γ = 0.99, three health levels, and four habituation levels. Exact paper settings remain unverified because the paper PDF is not in this workspace.</div>
      <div className="form-actions"><button className="button primary" disabled={busy}>{busy ? "Submitting…" : "Create experiment"}</button></div>
      {message && <div className="notice success">{message}<button className="text-button" onClick={() => navigate("/history")}> Open history →</button></div>}
      {error && <ErrorMessage message={error} />}
    </form>
  </>;
}

function History() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [comparison, setComparison] = useState<Experiment[] | null>(null);
  const [error, setError] = useState("");
  async function load() {
    try { const result = await api.experiments(); setExperiments(result.experiments); setError(""); }
    catch (e) { setError((e as Error).message); }
  }
  useEffect(() => { void load(); }, []);
  async function compare() {
    try { setComparison((await api.compare(selected)).experiments); setError(""); }
    catch (e) { setError((e as Error).message); }
  }
  async function exportRun(id: number, format: "csv" | "json") {
    try {
      const result = await api.export(id, format);
      const content = format === "csv" ? String(result.content) : JSON.stringify(result.content, null, 2);
      const url = URL.createObjectURL(new Blob([content], { type: format === "csv" ? "text/csv" : "application/json" }));
      const link = document.createElement("a"); link.href = url; link.download = result.filename; link.click(); URL.revokeObjectURL(url);
    } catch (e) { setError((e as Error).message); }
  }
  return <>
    <PageHeading kicker="EXPERIMENT ARCHIVE" title="History & comparison" text="Review persisted runs, inspect their results, and compare compatible experiment summaries." />
    <div className="toolbar"><span>{experiments.length} experiment{experiments.length === 1 ? "" : "s"}</span><button className="button secondary" onClick={() => void load()}>Refresh</button><button className="button primary" disabled={selected.length < 2} onClick={() => void compare()}>Compare selected ({selected.length})</button></div>
    {error && <ErrorMessage message={error} />}
    <div className="panel table-scroll"><table><thead><tr><th>Select</th><th>Name</th><th>Algorithm</th><th>State</th><th>Status</th><th>Mean return</th><th>Created</th><th>Actions</th></tr></thead><tbody>
      {experiments.map(item => <tr key={item.id}><td><input aria-label={`Select experiment ${item.id}`} type="checkbox" checked={selected.includes(item.id)} onChange={e => setSelected(e.target.checked ? [...selected, item.id] : selected.filter(id => id !== item.id))} /></td><td><a href={`/history/${item.id}`}>{item.name}</a></td><td>{item.algorithm}</td><td>{item.fatigue_aware ? "Fatigue-aware" : "Memoryless"}</td><td><span className={`status-pill ${item.status}`}>{item.status}</span></td><td>{item.mean_discounted_return?.toFixed(3) ?? "—"}</td><td>{item.created_at ? new Date(item.created_at).toLocaleString() : "—"}</td><td><button className="text-button" onClick={() => void exportRun(item.id, "csv")}>CSV</button> · <button className="text-button" onClick={() => void exportRun(item.id, "json")}>JSON</button></td></tr>)}
      {experiments.length === 0 && <tr><td colSpan={8} className="empty">No experiments yet. Configure one to populate your archive.</td></tr>}
    </tbody></table></div>
    {comparison && <section className="panel"><div className="section-title"><div><span className="eyebrow">COMPARISON</span><h2>Selected experiment returns</h2></div></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={comparison}><CartesianGrid strokeDasharray="3 3" stroke="#26344a" /><XAxis dataKey="algorithm" stroke="#8c9bb1" /><YAxis stroke="#8c9bb1" /><Tooltip /><Bar dataKey="mean_discounted_return" fill="#59c7b6" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></div></section>}
  </>;
}

function ExperimentDetailPage() {
  const { id } = useParams();
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [evaluation, setEvaluation] = useState<Record<string, unknown> | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!id) return;
    Promise.all([api.experiments(), api.results(Number(id))]).then(([history, results]) => {
      setExperiment(history.experiments.find(item => item.id === Number(id)) ?? null);
      setEvaluation(results.evaluation);
      setTrainingMetrics(results.training_metrics);
    }).catch((e: Error) => setError(e.message));
  }, [id]);
  if (error) return <ErrorMessage message={error} />;
  if (!experiment) return <div className="notice">Loading experiment results…</div>;
  const sample = evaluation?.sample_episode as { reward_sequence?: number[]; habituation_trajectory?: number[] } | null;
  const series = (sample?.reward_sequence ?? []).map((reward, i) => ({ day: i + 1, reward, habituation: sample?.habituation_trajectory?.[i + 1] ?? 0 }));
  const trainingReturns = (trainingMetrics?.episode_returns as number[] | undefined) ?? [];
  const trainingDeltas = (trainingMetrics?.value_deltas as number[] | undefined) ?? [];
  const trainingSeries = (trainingReturns.length ? trainingReturns : trainingDeltas).map((value, index) => ({ iteration: index + 1, value }));
  return <>
    <PageHeading kicker={`EXPERIMENT #${experiment.id}`} title={experiment.name} text={`${experiment.algorithm} · ${experiment.fatigue_aware ? "Fatigue-aware" : "Memoryless"} · ${experiment.status}`} />
    {experiment.status === "failed" && <ErrorMessage message={experiment.error_message ?? "Experiment failed without an error message."} />}
    <div className="metric-grid"><Metric label="Mean discounted return" value={experiment.mean_discounted_return?.toFixed(3) ?? "—"} detail={`95% CI ${experiment.ci_95_lower?.toFixed(3) ?? "—"} – ${experiment.ci_95_upper?.toFixed(3) ?? "—"}`} /><Metric label="Health score" value={experiment.mean_health_score?.toFixed(3) ?? "—"} detail="Episode-level aggregate" /><Metric label="Response rate" value={experiment.mean_response_rate?.toFixed(3) ?? "—"} detail="Among delivered nudges" /><Metric label="Mean habituation" value={experiment.mean_habituation?.toFixed(3) ?? "—"} detail="Across evaluation episodes" /></div>
    {series.length > 0 && <section className="panel"><span className="eyebrow">SAMPLE EPISODE</span><h2>Reward and habituation trace</h2><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={series}><CartesianGrid strokeDasharray="3 3" stroke="#26344a" /><XAxis dataKey="day" stroke="#8c9bb1" /><YAxis stroke="#8c9bb1" /><Tooltip /><Line dataKey="reward" stroke="#59c7b6" dot={false} /><Line dataKey="habituation" stroke="#d580a7" dot={false} /></LineChart></ResponsiveContainer></div></section>}
    {trainingSeries.length > 0 && <section className="panel"><span className="eyebrow">TRAINING TRACE</span><h2>{trainingReturns.length ? "Episode return" : "Value-iteration delta"}</h2><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={trainingSeries}><CartesianGrid strokeDasharray="3 3" stroke="#26344a" /><XAxis dataKey="iteration" stroke="#8c9bb1" /><YAxis stroke="#8c9bb1" /><Tooltip /><Line dataKey="value" stroke="#62a8ff" dot={false} /></LineChart></ResponsiveContainer></div></section>}
  </>;
}

function PolicyInspector() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [id, setId] = useState("");
  const [entries, setEntries] = useState<PolicyEntry[]>([]);
  const [error, setError] = useState("");
  useEffect(() => { api.experiments().then(value => setExperiments(value.experiments)).catch((e: Error) => setError(e.message)); }, []);
  async function inspect() {
    try { const data = await api.policy(Number(id)); setEntries(data.entries); setError(""); }
    catch (e) { setError((e as Error).message); }
  }
  return <>
    <PageHeading kicker="POLICY EXPLORER" title="Inspect state-to-action mappings" text="Browse the persisted action table for a completed policy and see its state-specific response probability." />
    <div className="toolbar"><label>Completed run<select value={id} onChange={e => setId(e.target.value)}><option value="">Select an experiment</option>{experiments.filter(item => item.status === "completed").map(item => <option key={item.id} value={item.id}>#{item.id} · {item.name}</option>)}</select></label><button className="button primary" disabled={!id} onClick={() => void inspect()}>Load policy</button></div>
    {error && <ErrorMessage message={error} />}
    {entries.length > 0 && <div className="panel table-scroll"><table><thead><tr><th>Sleep</th><th>Activity</th><th>Stress</th><th>Habituation</th><th>Action</th><th>Response probability</th></tr></thead><tbody>{entries.map((entry, index) => <tr key={`${entry.state.join("-")}-${index}`}><td>{entry.state[0]}</td><td>{entry.state[1]}</td><td>{entry.state[2]}</td><td>{entry.state[3] ?? "Unobserved"}</td><td>{entry.action_name}</td><td>{entry.response_probability.toFixed(3)}</td></tr>)}</tbody></table></div>}
  </>;
}

function Methodology() {
  return <>
    <PageHeading kicker="SCIENTIFIC CONTEXT" title="Methodology & limitations" text="The platform is a synthetic research simulator. Verify model assumptions against the source paper before treating an experiment as a reproduction." />
    <section className="panel"><span className="eyebrow">IMPLEMENTED MODEL DEFAULTS</span><h2>State and transition outline</h2><ul className="method-list"><li>Health state: sleep, activity, and stress use three ordinal levels; habituation uses four levels.</li><li>Actions: no nudge, sleep nudge, activity nudge, and stress-management nudge.</li><li>Response probability is a nudge-specific baseline adjusted by a habituation attenuation factor.</li><li>Responses may improve the targeted health dimension; each dimension may also experience natural downward drift.</li><li>Algorithms include rule-based, value-iteration DP, first-visit Monte Carlo control, tabular Q-learning, and full-state oracle DP.</li><li>Evaluation reports discounted return, health score, nudge frequency, response rate, habituation, and confidence intervals across replications.</li></ul></section>
    <p className="fine-print">The model uses synthetic transitions only. It contains no patient records and has not been clinically validated. Outputs must not be used for treatment recommendations.</p>
  </>;
}

export default App;
