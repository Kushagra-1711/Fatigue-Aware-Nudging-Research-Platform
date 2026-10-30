import time

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_research_api_workflow(tmp_path):
    database_url = f"sqlite:///{tmp_path.as_posix()}/research.db"
    client = TestClient(create_app(database_url))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    preflight = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"

    alternate_preflight = client.options(
        "/api/dashboard",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert alternate_preflight.status_code == 200

    defaults = client.get("/api/config/defaults")
    assert defaults.status_code == 200
    assert defaults.json()["simulator"]["episode_length"] == 28

    simulation = client.post(
        "/api/simulations",
        json={
            "algorithm": "rule_based",
            "seed": 13,
            "config": {"episode_length": 3},
        },
    )
    assert simulation.status_code == 200
    assert len(simulation.json()["steps"]) == 3

    invalid_config = client.post(
        "/api/simulations",
        json={"algorithm": "rule_based", "config": {"initial_sleep": 3}},
    )
    assert invalid_config.status_code == 422

    over_budget = client.post(
        "/api/experiments",
        json={
            "name": "Over budget",
            "algorithm": "rule_based",
            "simulator_config": {"episode_length": 365},
            "n_eval_episodes": 5000,
            "n_replications": 20,
        },
    )
    assert over_budget.status_code == 422

    created = client.post(
        "/api/experiments",
        json={
            "name": "API integration",
            "algorithm": "rule_based",
            "fatigue_aware": True,
            "n_eval_episodes": 10,
            "n_replications": 1,
            "base_seed": 5,
        },
    )
    assert created.status_code == 200
    experiment_id = created.json()["id"]

    for _ in range(100):
        experiment = client.get(f"/api/experiments/{experiment_id}").json()
        if experiment["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert experiment["status"] == "completed", experiment
    assert experiment["fatigue_aware"] is False
    assert experiment["created_at"].endswith("+00:00")
    results = client.get(f"/api/experiments/{experiment_id}/results")
    assert results.status_code == 200
    assert results.json()["evaluation"]["n_episodes_per_replication"] == 10

    policy = client.get(f"/api/experiments/{experiment_id}/policy")
    assert policy.status_code == 200
    assert policy.json()["total_states"] == 27

    export = client.get(f"/api/experiments/{experiment_id}/export?format=csv")
    assert export.status_code == 200
    assert export.json()["format"] == "csv"
    assert "mean_discounted_return" in export.json()["content"]

    history = client.get("/api/experiments")
    assert history.status_code == 200
    assert history.json()["total"] == 1
