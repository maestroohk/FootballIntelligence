using System.Globalization;

namespace FootballIntelligence.Web.Services;

public static class Fmt
{
    private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    public static string Pct(double? p, int digits = 0) => p is null || double.IsNaN(p.Value) ? "—" : (p.Value * 100).ToString("F" + digits, Inv) + "%";
    public static string Num(double? v, int digits = 2) => v is null || double.IsNaN(v.Value) ? "—" : v.Value.ToString("F" + digits, Inv);
    public static string Signed(double? v, int digits = 4) => v is null ? "—" : (v.Value >= 0 ? "+" : "") + v.Value.ToString("F" + digits, Inv);
    public static string Inv2(double v) => v.ToString("0.###", Inv);

    public static string Ago(string? iso)
    {
        if (string.IsNullOrEmpty(iso) || !DateTime.TryParse(iso, Inv, DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var t)) return "never";
        var d = DateTime.UtcNow - t;
        if (d.TotalSeconds < 0) return "just now";
        if (d.TotalSeconds < 60) return $"{(int)d.TotalSeconds}s ago";
        if (d.TotalMinutes < 60) return $"{(int)d.TotalMinutes}m ago";
        if (d.TotalHours < 48) return $"{(int)d.TotalHours}h ago";
        return $"{(int)d.TotalDays}d ago";
    }

    public static TimeSpan? Since(string? iso) =>
        string.IsNullOrEmpty(iso) || !DateTime.TryParse(iso, Inv, DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var t) ? null : DateTime.UtcNow - t;

    public static string Duration(TimeSpan? d)
    {
        if (d is null) return "—";
        var v = d.Value;
        if (v.TotalSeconds < 60) return $"{(int)v.TotalSeconds}s";
        if (v.TotalMinutes < 60) return $"{(int)v.TotalMinutes}m {v.Seconds}s";
        return $"{(int)v.TotalHours}h {v.Minutes}m";
    }

    public static string Local(string? iso) =>
        string.IsNullOrEmpty(iso) || !DateTime.TryParse(iso, Inv, DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var t) ? "—" : t.ToLocalTime().ToString("dd MMM HH:mm:ss", Inv);

    public static string Initials(string name)
    {
        var parts = name.Replace("'", "").Split(' ', StringSplitOptions.RemoveEmptyEntries);
        return parts.Length >= 2 ? $"{parts[0][0]}{parts[1][0]}".ToUpperInvariant() : name.Length >= 3 ? name[..3].ToUpperInvariant() : name.ToUpperInvariant();
    }

    public static string TeamColor(string name)
    {
        var h = 0;
        foreach (var c in name) h = (h * 31 + c) & 0x7fffffff;
        var hue = h % 360;
        return $"hsl({hue},45%,38%)";
    }

    public static string Minute(double clock, long period = 0)
    {
        var m = (int)Math.Floor(clock) + 1;
        if (period == 1 && m > 45) return $"45+{m - 45}'";
        if (period == 2 && m > 90) return $"90+{m - 90}'";
        return $"{m}'";
    }

    public static string CompetitionName(string code) => code switch
    {
        "E0" => "Premier League", "E1" => "Championship", "E2" => "League One", "SP1" => "La Liga", "SP2" => "Segunda",
        "D1" => "Bundesliga", "D2" => "2. Bundesliga", "I1" => "Serie A", "I2" => "Serie B", "F1" => "Ligue 1", "F2" => "Ligue 2",
        "N1" => "Eredivisie", "P1" => "Primeira Liga", "B1" => "Pro League", "SC0" => "Scottish Premiership", "SC1" => "Scottish Championship",
        "T1" => "Süper Lig", "G1" => "Super League Greece", _ => code
    };

    public static string Country(string code) => code switch
    {
        "E0" or "E1" or "E2" => "ENG", "SP1" or "SP2" => "ESP", "D1" or "D2" => "GER", "I1" or "I2" => "ITA", "F1" or "F2" => "FRA",
        "N1" => "NED", "P1" => "POR", "B1" => "BEL", "SC0" or "SC1" => "SCO", "T1" => "TUR", "G1" => "GRE", _ => ""
    };

    public static string LeagueColor(string code) => code switch
    {
        "E0" => "#7b3fe4", "E1" or "E2" => "#4a5bd6", "SP1" or "SP2" => "#e2483d", "D1" or "D2" => "#d6262f", "I1" or "I2" => "#1d6fe0",
        "F1" or "F2" => "#2f6bd8", "N1" => "#f07c1e", "P1" => "#1f9e57", "B1" => "#c9a227", "SC0" or "SC1" => "#2758a6", "T1" => "#cf2236", "G1" => "#2f73c9", _ => "#5b6b80"
    };

    public static (string Primary, string Secondary) TeamColors(string name)
    {
        var h = 0;
        foreach (var c in name) h = (h * 31 + c) & 0x7fffffff;
        var hue = h % 360;
        return ($"hsl({hue},55%,42%)", $"hsl({(hue + 140) % 360},45%,88%)");
    }

    public static string ModelLabel(string? version) => version is null ? "—" : version.Split('@')[0] switch
    {
        "dixon_coles" => "Dixon-Coles", "poisson" => "Poisson (Maher)", "frequency" => "Frequency baseline", "elo" => "Elo (K=20)",
        "elo_k15" => "Elo (K=15)", "elo_k30" => "Elo (K=30)", "dc_xi0010" => "Dixon-Coles slow decay", "dc_xi0030" => "Dixon-Coles fast decay",
        "dc_ridge3" => "Dixon-Coles strong ridge", "dc_xg25" => "DC + shot-xG 25%", "dc_xg50" => "DC + shot-xG 50%", "dc_xg75" => "DC + shot-xG 75%",
        "gbm_stacked" => "LightGBM stacked", "gbm_features" => "LightGBM features", "logistic" => "Multinomial logistic",
        "ensemble" => "Learned ensemble", "bookmaker_close" => "Bookmaker closing (benchmark)", var other => other
    };
}
