using System.Text.Json;

namespace FootballIntelligence.Web.Services;

public class Markets
{
    public double PHome { get; init; }
    public double PDraw { get; init; }
    public double PAway { get; init; }
    public double ExpHome { get; init; }
    public double ExpAway { get; init; }
    public double Btts { get; init; }
    public double PHomeScores { get; init; }
    public double PAwayScores { get; init; }
    public double HomeCleanSheet { get; init; }
    public double AwayCleanSheet { get; init; }
    public double PGoalFirstHalf { get; init; }
    public double PGoalSecondHalf { get; init; }
    public double PGoalEachHalf { get; init; }
    public double OutcomeEntropy { get; init; }
    public Dictionary<string, double> Over { get; init; } = [];
    public List<(int Home, int Away, double P)> TopScores { get; init; } = [];
    public double[][] Matrix { get; init; } = [];
    public double[] TotalGoals { get; init; } = [];
    public double ExpTotal => ExpHome + ExpAway;
    public double ExpDiff => ExpHome - ExpAway;

    public static Markets? FromJson(string? json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;
        var root = JsonDocument.Parse(json).RootElement;
        if (!root.TryGetProperty("p_home", out _)) return null;
        double D(string k) => root.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : double.NaN;
        var over = new Dictionary<string, double>();
        if (root.TryGetProperty("over_under", out var ou))
            foreach (var p in ou.EnumerateObject()) over[p.Name] = p.Value.GetProperty("over").GetDouble();
        var top = root.TryGetProperty("top_scores", out var ts)
            ? ts.EnumerateArray().Select(x => (x.GetProperty("home").GetInt32(), x.GetProperty("away").GetInt32(), x.GetProperty("p").GetDouble())).ToList()
            : [];
        var matrix = root.TryGetProperty("matrix", out var mx) ? mx.EnumerateArray().Select(r => r.EnumerateArray().Select(c => c.GetDouble()).ToArray()).ToArray() : [];
        var totals = root.TryGetProperty("total_goals", out var tg) ? tg.EnumerateArray().Select(x => x.GetDouble()).ToArray() : [];
        return new Markets
        {
            PHome = D("p_home"), PDraw = D("p_draw"), PAway = D("p_away"), ExpHome = D("exp_home"), ExpAway = D("exp_away"),
            Btts = D("btts"), PHomeScores = D("p_home_scores"), PAwayScores = D("p_away_scores"),
            HomeCleanSheet = D("home_clean_sheet"), AwayCleanSheet = D("away_clean_sheet"),
            PGoalFirstHalf = D("p_goal_first_half"), PGoalSecondHalf = D("p_goal_second_half"), PGoalEachHalf = D("p_goal_each_half"),
            OutcomeEntropy = D("outcome_entropy"), Over = over, TopScores = top, Matrix = matrix, TotalGoals = totals,
        };
    }
}

public static class ScoreModel
{
    public const int MaxGoals = 10;
    public const double FirstHalfShare = 0.445;

    public static double[][] DixonColes(double lh, double la, double rho)
    {
        var ph = Pmf(lh);
        var pa = Pmf(la);
        var m = new double[MaxGoals + 1][];
        for (var i = 0; i <= MaxGoals; i++)
        {
            m[i] = new double[MaxGoals + 1];
            for (var j = 0; j <= MaxGoals; j++) m[i][j] = ph[i] * pa[j];
        }
        m[0][0] *= 1 - lh * la * rho;
        m[0][1] *= 1 + lh * rho;
        m[1][0] *= 1 + la * rho;
        m[1][1] *= 1 - rho;
        var total = m.Sum(r => r.Sum(x => Math.Max(0, x)));
        for (var i = 0; i <= MaxGoals; i++)
            for (var j = 0; j <= MaxGoals; j++) m[i][j] = Math.Max(0, m[i][j]) / total;
        return m;
    }

    private static double[] Pmf(double lambda)
    {
        var p = new double[MaxGoals + 1];
        p[0] = Math.Exp(-lambda);
        for (var k = 1; k <= MaxGoals; k++) p[k] = p[k - 1] * lambda / k;
        return p;
    }

    public static Markets Compute(double[][] m)
    {
        int n = m.Length;
        double ph = 0, pd = 0, pa = 0, eh = 0, ea = 0;
        var totals = new double[2 * n - 1];
        var scores = new List<(int, int, double)>();
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                var p = m[i][j];
                if (i > j) ph += p; else if (i == j) pd += p; else pa += p;
                eh += i * p;
                ea += j * p;
                totals[i + j] += p;
                scores.Add((i, j, p));
            }
        double colZero = 0, rowZero = 0, btts = 0;
        for (var k = 0; k < n; k++) { colZero += m[k][0]; rowZero += m[0][k]; }
        for (var i = 1; i < n; i++) for (var j = 1; j < n; j++) btts += m[i][j];
        double first = 0, second = 0, each = 0;
        for (var t = 0; t < totals.Length; t++)
        {
            first += totals[t] * (1 - Math.Pow(1 - FirstHalfShare, t));
            second += totals[t] * (1 - Math.Pow(FirstHalfShare, t));
            if (t > 0) each += totals[t] * (1 - Math.Pow(FirstHalfShare, t) - Math.Pow(1 - FirstHalfShare, t));
        }
        var over = new Dictionary<string, double>();
        foreach (var line in new[] { 0.5, 1.5, 2.5, 3.5, 4.5 })
            over[line.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture)] = totals.Where((_, t) => t > line).Sum();
        var probs = new[] { ph, pd, pa };
        return new Markets
        {
            PHome = ph, PDraw = pd, PAway = pa, ExpHome = eh, ExpAway = ea,
            PHomeScores = 1 - rowZero, PAwayScores = 1 - colZero, Btts = btts,
            HomeCleanSheet = colZero, AwayCleanSheet = rowZero,
            PGoalFirstHalf = first, PGoalSecondHalf = second, PGoalEachHalf = each,
            OutcomeEntropy = -probs.Sum(p => p > 0 ? p * Math.Log(p) : 0),
            Over = over,
            TopScores = scores.OrderByDescending(s => s.Item3).Take(8).ToList(),
            Matrix = m.Take(7).Select(r => r.Take(7).ToArray()).ToArray(),
            TotalGoals = totals.Take(9).ToArray(),
        };
    }
}
