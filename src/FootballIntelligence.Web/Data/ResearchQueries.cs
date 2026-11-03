using System.Text.Json;

namespace FootballIntelligence.Web.Data;

public class ResearchQueries(Database db)
{
    public List<BacktestRow> Backtests(string split) => db.Query<BacktestRow>(
        "SELECT * FROM backtest_runs WHERE split=@split AND run_id IN (SELECT MAX(run_id) FROM backtest_runs WHERE split=@split GROUP BY model_version) ORDER BY json_extract(metrics,'$.log_loss')",
        new { split }).ToList();

    public BacktestRow? Backtest(string model, string split) => db.Single<BacktestRow>(
        "SELECT * FROM backtest_runs WHERE model_version=@model AND split=@split ORDER BY run_id DESC LIMIT 1", new { model, split });

    public List<ExperimentRow> Experiments() => db.Query<ExperimentRow>("SELECT * FROM experiments ORDER BY experiment_id DESC").ToList();

    public List<ModelVersionRow> Versions() => db.Query<ModelVersionRow>("SELECT * FROM model_versions ORDER BY CASE status WHEN 'champion' THEN 0 WHEN 'benchmark' THEN 2 ELSE 1 END, version").ToList();

    public string? MostPredictedModel(string purpose) => db.Single<string>(
        "SELECT model_version FROM predictions WHERE purpose=@purpose AND model_version NOT LIKE 'bookmaker%' GROUP BY model_version ORDER BY COUNT(*) DESC LIMIT 1", new { purpose });

    public string? Champion() => db.Single<string>("SELECT version FROM model_versions WHERE status='champion' LIMIT 1");

    public JsonElement? KvJson(string key)
    {
        var raw = db.Kv(key);
        if (raw is null) return null;
        using var doc = JsonDocument.Parse(raw);
        return doc.RootElement.Clone();
    }

    public List<(string Name, JsonElement Value)> Comparisons()
    {
        var root = KvJson("validation_comparisons");
        if (root is null) return [];
        return root.Value.EnumerateObject().Select(p => (p.Name, p.Value)).ToList();
    }

    public List<(string Name, JsonElement Value)> KvComparisons(string key)
    {
        var root = KvJson(key);
        if (root is null) return [];
        return root.Value.EnumerateObject().Select(p => (p.Name, p.Value)).ToList();
    }

    public List<(string Name, JsonElement Value)> FinalComparisons()
    {
        var root = KvJson("final_test_comparisons");
        if (root is null) return [];
        return root.Value.EnumerateObject().Select(p => (p.Name, p.Value)).ToList();
    }

    public List<PerformancePoint> RollingPerformance(string model, string purpose)
    {
        var rows = db.Query<SettledRow>("""
            SELECT substr(m.match_date,1,7) period, p.p_home, p.p_draw, p.p_away, p.lambda_home, p.lambda_away, m.fthg, m.ftag
            FROM predictions p JOIN matches m ON m.match_id=p.match_id
            WHERE p.model_version=@model AND p.purpose=@purpose AND m.status='finished'
              AND m.competition IN ('E0','E1','SP1','D1','I1','F1')
            """, new { model, purpose });
        return rows.GroupBy(r => r.Period).OrderBy(g => g.Key).Select(g =>
        {
            var list = g.ToList();
            var withGoals = list.Where(r => r.LambdaHome is not null).ToList();
            return new PerformancePoint
            {
                Period = g.Key,
                N = list.Count,
                LogLoss = list.Average(r => -Math.Log(Math.Max(1e-12, r.PActual))),
                Accuracy = list.Average(r => r.TopHit ? 1.0 : 0.0),
                GoalMae = withGoals.Count == 0 ? null : withGoals.Average(r => (Math.Abs(r.LambdaHome!.Value - r.Fthg) + Math.Abs(r.LambdaAway!.Value - r.Ftag)) / 2),
            };
        }).ToList();
    }

    public List<GoalRow> GoalRows(string model, string purpose) => db.Query<GoalRow>("""
        SELECT p.lambda_home, p.lambda_away, COALESCE(json_extract(p.markets, '$.rho'), 0.0) rho, m.fthg, m.ftag
        FROM predictions p JOIN matches m ON m.match_id=p.match_id
        WHERE p.model_version=@model AND p.purpose=@purpose AND m.status='finished' AND p.lambda_home IS NOT NULL
          AND m.competition IN ('E0','E1','SP1','D1','I1','F1')
        """, new { model, purpose }).ToList();

    public List<MatchRow> PredictionHistory(string? model, string purpose, int limit) => db.Query<MatchRow>($"""
        SELECT m.match_id, m.competition, m.season, m.match_date, m.status, {Database.DisplayName("th.team_id", "th.name")} home_name, {Database.DisplayName("ta.team_id", "ta.name")} away_name, m.fthg, m.ftag,
               p.p_home, p.p_draw, p.p_away, p.lambda_home, p.lambda_away, p.model_version, p.purpose, p.created_at prediction_created, p.data_cutoff
        FROM predictions p JOIN matches m ON m.match_id=p.match_id
        JOIN teams th ON th.team_id=m.home_team_id JOIN teams ta ON ta.team_id=m.away_team_id
        WHERE (@model IS NULL OR p.model_version=@model) AND p.purpose=@purpose
        ORDER BY m.match_date DESC, m.match_id DESC LIMIT @limit
        """, new { model, purpose, limit }).ToList();

    public List<LiveModelRow> LiveModels() => db.Query<LiveModelRow>("SELECT version, created_at, params, metrics, status FROM live_model ORDER BY created_at DESC").ToList();
}

public class PerformancePoint
{
    public string Period { get; set; } = "";
    public long N { get; set; }
    public double LogLoss { get; set; }
    public double Accuracy { get; set; }
    public double? GoalMae { get; set; }
}

public class LiveModelRow
{
    public string Version { get; set; } = "";
    public string CreatedAt { get; set; } = "";
    public string Params { get; set; } = "";
    public string? Metrics { get; set; }
    public string Status { get; set; } = "";
}

public class SettledRow
{
    public string Period { get; set; } = "";
    public double PHome { get; set; }
    public double PDraw { get; set; }
    public double PAway { get; set; }
    public double? LambdaHome { get; set; }
    public double? LambdaAway { get; set; }
    public long Fthg { get; set; }
    public long Ftag { get; set; }
    public double PActual => Fthg > Ftag ? PHome : Fthg == Ftag ? PDraw : PAway;
    public bool TopHit
    {
        get
        {
            var top = PHome >= PDraw && PHome >= PAway ? 0 : PDraw >= PAway ? 1 : 2;
            var actual = Fthg > Ftag ? 0 : Fthg == Ftag ? 1 : 2;
            return top == actual;
        }
    }
}

public class GoalRow
{
    public double LambdaHome { get; set; }
    public double LambdaAway { get; set; }
    public double Rho { get; set; }
    public long Fthg { get; set; }
    public long Ftag { get; set; }
}
