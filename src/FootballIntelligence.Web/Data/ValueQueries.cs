using System.Text.Json;
using FootballIntelligence.Web.Services;

namespace FootballIntelligence.Web.Data;

public class ValueQueries(Database db)
{
    private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };

    private T? Kv<T>(string key) where T : class
    {
        var raw = db.Kv(key);
        return raw is null ? null : JsonSerializer.Deserialize<T>(raw, Json);
    }

    public ValueBoard? Board() => Kv<ValueBoard>("value_board");

    public StrategyResults? Strategy() => Kv<StrategyResults>("strategy_research");

    public List<PaperDecisionRow> PaperDecisions(int limit = 200) => db.Query<PaperDecisionRow>($"""
        SELECT d.decision_id, d.created_at, d.match_id, d.rule, d.decision, d.reason, d.market, d.line, d.selection, d.bookmaker, d.odds,
               d.odds_observed_at, d.model_prob, d.market_prob, d.ev, d.stake, m.match_date, m.kickoff, m.competition,
               {Database.DisplayName("m.home_team_id", "th.name")} home_name, {Database.DisplayName("m.away_team_id", "ta.name")} away_name,
               s.home_goals, s.away_goals, s.profit, s.settled_at
        FROM paper_decisions d JOIN matches m ON m.match_id = d.match_id
        JOIN teams th ON th.team_id = m.home_team_id JOIN teams ta ON ta.team_id = m.away_team_id
        LEFT JOIN paper_settlements s ON s.decision_id = d.decision_id
        ORDER BY d.decision = 'bet' DESC, m.match_date DESC, d.ev DESC LIMIT @limit
        """, new { limit }).ToList();

    public bool PaperExists() => db.Single<long>("SELECT COUNT(*) FROM sqlite_master WHERE name = 'paper_decisions'") > 0;

    public List<ExperimentRow> MarketExperiments() => db.Query<ExperimentRow>("""
        SELECT * FROM experiments
        WHERE name LIKE '%bet%' OR name LIKE '%market%' OR name LIKE '%bookmaker%' OR name LIKE '%favourite%' OR name LIKE '%calibration%' OR name LIKE '%price%'
        ORDER BY experiment_id DESC
        """).ToList();
}

public class ValueBoard
{
    public string GeneratedAt { get; set; } = "";
    public List<BoardMatch> Matches { get; set; } = [];
    public List<PriceCoverage> Coverage { get; set; } = [];
}

public class PriceCoverage
{
    public string Bookmaker { get; set; } = "";
    public string Timing { get; set; } = "";
    public string Kind { get; set; } = "";
    public long Prices { get; set; }
    public long Matches { get; set; }
    public string? Latest { get; set; }
}

public class BoardMatch
{
    public long MatchId { get; set; }
    public string Date { get; set; } = "";
    public string? Kickoff { get; set; }
    public string Competition { get; set; } = "";
    public string Home { get; set; } = "";
    public string Away { get; set; } = "";
    public string ModelVersion { get; set; } = "";
    public string PredictedAt { get; set; } = "";
    public bool SafelyBeforeKickoff { get; set; }
    public List<BoardMarket> Markets { get; set; } = [];
}

public class BoardMarket
{
    public string Market { get; set; } = "";
    public double Line { get; set; }
    public string Bookmaker { get; set; } = "";
    public string ObservedAt { get; set; } = "";
    public double Margin { get; set; }
    public List<BoardSelection> Selections { get; set; } = [];
}

public class BoardSelection
{
    public string Selection { get; set; } = "";
    public double Odds { get; set; }
    public double ModelWin { get; set; }
    public double ModelPush { get; set; }
    public double Implied { get; set; }
    public double MarketProb { get; set; }
    public double BreakEvenOdds { get; set; }
    public double Ev { get; set; }
}

public class StrategyResults
{
    public Dictionary<string, List<int>> Periods { get; set; } = [];
    public List<StrategyRule> Rules { get; set; } = [];
    public Dictionary<string, StrategySummary> Baselines { get; set; } = [];
    public Dictionary<string, ProbabilityQuality> ProbabilityQuality { get; set; } = [];
}

public class StrategyRule
{
    public string Market { get; set; } = "";
    public string Bookmaker { get; set; } = "";
    public RuleParams? Selected { get; set; }
    public string? Reason { get; set; }
    public StrategySummary? Development { get; set; }
    public StrategySummary? Holdout { get; set; }
    public StrategySummary? Final { get; set; }
    public List<GridRow> Grid { get; set; } = [];
    public List<EquityPoint> EquityHoldoutFinal { get; set; } = [];
}

public class RuleParams
{
    public double MinEv { get; set; }
    public double OddsMin { get; set; }
    public double OddsMax { get; set; }
}

public class GridRow
{
    public double MinEv { get; set; }
    public double OddsMin { get; set; }
    public double OddsMax { get; set; }
    public long N { get; set; }
    public double Roi { get; set; }
}

public class EquityPoint
{
    public string Date { get; set; } = "";
    public double Units { get; set; }
}

public class StrategySummary
{
    public long N { get; set; }
    public double Profit { get; set; }
    public double? Roi { get; set; }
    public List<double>? RoiCi { get; set; }
    public double MaxDrawdown { get; set; }
    public long LongestLosingStreak { get; set; }
    public double AvgOdds { get; set; }
    public double MeanModelEv { get; set; }
    public double WinRate { get; set; }
    public double? ClvMean { get; set; }
    public double? ClvBeatShare { get; set; }
    public long ClvN { get; set; }
    public List<int> Seasons { get; set; } = [];
}

public class ProbabilityQuality
{
    public long N { get; set; }
    public double ModelLogLoss { get; set; }
    public double PinnacleOpenLogLoss { get; set; }
    public double ModelBrier { get; set; }
    public double PinnacleOpenBrier { get; set; }
}

public class PaperDecisionRow
{
    public long DecisionId { get; set; }
    public string CreatedAt { get; set; } = "";
    public long MatchId { get; set; }
    public string Rule { get; set; } = "";
    public string Decision { get; set; } = "";
    public string Reason { get; set; } = "";
    public string? Market { get; set; }
    public double? Line { get; set; }
    public string? Selection { get; set; }
    public string? Bookmaker { get; set; }
    public double? Odds { get; set; }
    public string? OddsObservedAt { get; set; }
    public double? ModelProb { get; set; }
    public double? MarketProb { get; set; }
    public double? Ev { get; set; }
    public double Stake { get; set; }
    public string MatchDate { get; set; } = "";
    public string? Kickoff { get; set; }
    public string Competition { get; set; } = "";
    public string HomeName { get; set; } = "";
    public string AwayName { get; set; } = "";
    public long? HomeGoals { get; set; }
    public long? AwayGoals { get; set; }
    public double? Profit { get; set; }
    public string? SettledAt { get; set; }
}

public static class ValueFmt
{
    public static string Book(string? b) => b switch
    {
        "pinnacle" => "Pinnacle",
        "bet365" => "Bet365",
        "market_avg" => "Market average",
        "market_max" => "Best available",
        _ => b ?? "—",
    };

    public static string Market(string? m) => m switch { "1x2" => "Match result", "ou" => "Total goals", "ah" => "Asian handicap", _ => m ?? "—" };

    public static string Pick(string? market, double? line, string? selection, string home = "Home", string away = "Away")
    {
        var l = line ?? 0;
        return market switch
        {
            "1x2" => selection switch { "home" => home, "away" => away, _ => "Draw" },
            "ou" => $"{(selection == "over" ? "Over" : "Under")} {l.ToString("0.##", System.Globalization.CultureInfo.InvariantCulture)}",
            "ah" => selection == "home" ? $"{home} {Signed(l)}" : $"{away} {Signed(-l)}",
            _ => selection ?? "—",
        };
    }

    private static string Signed(double v) => v == 0 ? "0" : (v > 0 ? "+" : "−") + Math.Abs(v).ToString("0.##", System.Globalization.CultureInfo.InvariantCulture);

    public static string Ci(List<double>? ci) => ci is { Count: 2 } ? $"[{Fmt.Pct(ci[0], 1)}, {Fmt.Pct(ci[1], 1)}]" : "";
}
