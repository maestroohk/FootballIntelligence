using System.Text;

namespace FootballIntelligence.Web.Services;

public static class TeamIdentity
{
    private static readonly Dictionary<string, (string, string)> Colors = new()
    {
        ["arsenal"] = ("#db0007", "#ffffff"), ["aston villa"] = ("#670e36", "#95bfe5"), ["bournemouth"] = ("#da291c", "#000000"),
        ["brentford"] = ("#e30613", "#ffffff"), ["brighton"] = ("#0057b8", "#ffffff"), ["brighton and hove albion"] = ("#0057b8", "#ffffff"),
        ["burnley"] = ("#6c1d45", "#99d6ea"), ["chelsea"] = ("#034694", "#ffffff"), ["crystal palace"] = ("#1b458f", "#c4122e"),
        ["everton"] = ("#003399", "#ffffff"), ["fulham"] = ("#000000", "#ffffff"), ["ipswich"] = ("#0044a9", "#ffffff"), ["ipswich town"] = ("#0044a9", "#ffffff"),
        ["leeds"] = ("#ffcd00", "#1d428a"), ["leeds united"] = ("#ffcd00", "#1d428a"), ["leicester"] = ("#003090", "#fdbe11"), ["leicester city"] = ("#003090", "#fdbe11"),
        ["liverpool"] = ("#c8102e", "#00b2a9"), ["manchester city"] = ("#6cabdd", "#1c2c5b"), ["man city"] = ("#6cabdd", "#1c2c5b"),
        ["manchester united"] = ("#da291c", "#fbe122"), ["man united"] = ("#da291c", "#fbe122"), ["newcastle"] = ("#241f20", "#ffffff"), ["newcastle united"] = ("#241f20", "#ffffff"),
        ["nottingham forest"] = ("#dd0000", "#ffffff"), ["nottm forest"] = ("#dd0000", "#ffffff"), ["southampton"] = ("#d71920", "#ffffff"),
        ["sunderland"] = ("#eb172b", "#ffffff"), ["tottenham"] = ("#132257", "#ffffff"), ["tottenham hotspur"] = ("#132257", "#ffffff"),
        ["west ham"] = ("#7a263a", "#1bb1e7"), ["west ham united"] = ("#7a263a", "#1bb1e7"), ["wolves"] = ("#fdb913", "#231f20"),
        ["wolverhampton wanderers"] = ("#fdb913", "#231f20"), ["wolverhampton"] = ("#fdb913", "#231f20"), ["watford"] = ("#fbee23", "#ed2127"),
        ["norwich"] = ("#00a650", "#fff200"), ["norwich city"] = ("#00a650", "#fff200"), ["sheffield united"] = ("#ee2737", "#000000"), ["sheffield utd"] = ("#ee2737", "#000000"),
        ["west brom"] = ("#122f67", "#ffffff"), ["west bromwich albion"] = ("#122f67", "#ffffff"), ["stoke"] = ("#e03a3e", "#ffffff"), ["stoke city"] = ("#e03a3e", "#ffffff"),
        ["swansea"] = ("#121212", "#ffffff"), ["swansea city"] = ("#121212", "#ffffff"), ["cardiff"] = ("#0070b5", "#d1001f"), ["cardiff city"] = ("#0070b5", "#d1001f"),
        ["hull"] = ("#f5a12d", "#000000"), ["hull city"] = ("#f5a12d", "#000000"), ["middlesbrough"] = ("#e11b22", "#ffffff"), ["luton"] = ("#f78f1e", "#002d62"), ["luton town"] = ("#f78f1e", "#002d62"),
        ["derby"] = ("#ffffff", "#000000"), ["derby county"] = ("#ffffff", "#000000"), ["coventry"] = ("#59cbe8", "#ffffff"), ["coventry city"] = ("#59cbe8", "#ffffff"),
        ["blackburn"] = ("#009ee0", "#ffffff"), ["blackburn rovers"] = ("#009ee0", "#ffffff"), ["bolton"] = ("#ffffff", "#263c7e"), ["bolton wanderers"] = ("#ffffff", "#263c7e"),
        ["birmingham"] = ("#0000ff", "#ffffff"), ["birmingham city"] = ("#0000ff", "#ffffff"), ["qpr"] = ("#1d5ba4", "#ffffff"), ["queens park rangers"] = ("#1d5ba4", "#ffffff"),
        ["reading"] = ("#004494", "#ffffff"), ["wigan"] = ("#1d59af", "#ffffff"), ["wigan athletic"] = ("#1d59af", "#ffffff"), ["portsmouth"] = ("#001489", "#ffffff"),
        ["millwall"] = ("#001d5e", "#ffffff"), ["sheffield wednesday"] = ("#0e00f7", "#ffffff"), ["sheffield weds"] = ("#0e00f7", "#ffffff"), ["preston"] = ("#ffffff", "#001d4f"),
        ["bristol city"] = ("#e21b23", "#ffffff"), ["charlton"] = ("#d4021d", "#ffffff"), ["plymouth"] = ("#00563f", "#ffffff"), ["huddersfield"] = ("#0e63ad", "#ffffff"),
        ["real madrid"] = ("#febe10", "#00529f"), ["barcelona"] = ("#a50044", "#004d98"), ["atletico madrid"] = ("#cb3524", "#272e61"), ["ath madrid"] = ("#cb3524", "#272e61"),
        ["atletico de madrid"] = ("#cb3524", "#272e61"), ["sevilla"] = ("#d71920", "#ffffff"), ["real betis"] = ("#00954c", "#ffffff"), ["betis"] = ("#00954c", "#ffffff"),
        ["valencia"] = ("#ee3524", "#000000"), ["villarreal"] = ("#ffe667", "#005187"), ["athletic club"] = ("#ee2523", "#ffffff"), ["ath bilbao"] = ("#ee2523", "#ffffff"),
        ["athletic bilbao"] = ("#ee2523", "#ffffff"), ["real sociedad"] = ("#0067b1", "#ffffff"), ["sociedad"] = ("#0067b1", "#ffffff"), ["celta vigo"] = ("#8ac3ee", "#e5254e"), ["celta"] = ("#8ac3ee", "#e5254e"),
        ["espanyol"] = ("#007fc8", "#ffffff"), ["espanol"] = ("#007fc8", "#ffffff"), ["getafe"] = ("#005999", "#ffffff"), ["osasuna"] = ("#d91a21", "#0a346f"),
        ["mallorca"] = ("#e20613", "#000000"), ["girona"] = ("#cd2534", "#ffffff"), ["rayo vallecano"] = ("#e53027", "#ffffff"), ["vallecano"] = ("#e53027", "#ffffff"),
        ["alaves"] = ("#0761af", "#ffffff"), ["las palmas"] = ("#ffe400", "#0055a5"), ["granada"] = ("#c31632", "#ffffff"), ["levante"] = ("#b4053f", "#004c97"),
        ["eibar"] = ("#a50044", "#004d98"), ["leganes"] = ("#005bab", "#ffffff"), ["valladolid"] = ("#921b88", "#ffffff"), ["cadiz"] = ("#fde607", "#0045a7"), ["elche"] = ("#ffffff", "#05642c"),
        ["deportivo la coruna"] = ("#0061ad", "#ffffff"), ["la coruna"] = ("#0061ad", "#ffffff"), ["malaga"] = ("#0091d7", "#ffffff"), ["oviedo"] = ("#0055a5", "#ffffff"),
        ["bayern munich"] = ("#dc052d", "#0066b2"), ["bayern munchen"] = ("#dc052d", "#0066b2"), ["borussia dortmund"] = ("#fde100", "#000000"), ["dortmund"] = ("#fde100", "#000000"),
        ["rb leipzig"] = ("#dd0741", "#ffffff"), ["leipzig"] = ("#dd0741", "#ffffff"), ["bayer leverkusen"] = ("#e32221", "#000000"), ["leverkusen"] = ("#e32221", "#000000"),
        ["eintracht frankfurt"] = ("#e1000f", "#000000"), ["ein frankfurt"] = ("#e1000f", "#000000"), ["vfb stuttgart"] = ("#ffffff", "#e32219"), ["stuttgart"] = ("#ffffff", "#e32219"),
        ["vfl wolfsburg"] = ("#65b32e", "#ffffff"), ["wolfsburg"] = ("#65b32e", "#ffffff"), ["borussia monchengladbach"] = ("#000000", "#ffffff"), ["m gladbach"] = ("#000000", "#ffffff"),
        ["borussia mönchengladbach"] = ("#000000", "#ffffff"), ["werder bremen"] = ("#1d9053", "#ffffff"), ["sc freiburg"] = ("#e2001a", "#000000"), ["freiburg"] = ("#e2001a", "#000000"),
        ["union berlin"] = ("#eb1923", "#ffffff"), ["hoffenheim"] = ("#1961b5", "#ffffff"), ["mainz"] = ("#c3141e", "#ffffff"), ["mainz 05"] = ("#c3141e", "#ffffff"),
        ["augsburg"] = ("#ba3733", "#46714d"), ["koln"] = ("#ed1c24", "#ffffff"), ["fc koln"] = ("#ed1c24", "#ffffff"), ["köln"] = ("#ed1c24", "#ffffff"), ["hertha"] = ("#005ca9", "#ffffff"), ["hertha berlin"] = ("#005ca9", "#ffffff"),
        ["schalke 04"] = ("#004d9d", "#ffffff"), ["schalke"] = ("#004d9d", "#ffffff"), ["hamburg"] = ("#0a3f86", "#ffffff"), ["hamburger sv"] = ("#0a3f86", "#ffffff"),
        ["heidenheim"] = ("#e30613", "#003b79"), ["st pauli"] = ("#624839", "#ffffff"), ["bochum"] = ("#005ca9", "#ffffff"), ["hannover"] = ("#009a3d", "#000000"),
        ["juventus"] = ("#000000", "#ffffff"), ["inter"] = ("#010e80", "#000000"), ["inter milan"] = ("#010e80", "#000000"), ["internazionale"] = ("#010e80", "#000000"),
        ["ac milan"] = ("#fb090b", "#000000"), ["milan"] = ("#fb090b", "#000000"), ["napoli"] = ("#12a0d7", "#ffffff"), ["roma"] = ("#8e1f2f", "#f0bc42"), ["as roma"] = ("#8e1f2f", "#f0bc42"),
        ["lazio"] = ("#87d8f7", "#ffffff"), ["atalanta"] = ("#1e71b8", "#000000"), ["fiorentina"] = ("#482e92", "#ffffff"), ["torino"] = ("#8a1e03", "#ffffff"),
        ["bologna"] = ("#1a2f48", "#a21c26"), ["genoa"] = ("#ad1919", "#002350"), ["sampdoria"] = ("#1b5497", "#ffffff"), ["udinese"] = ("#000000", "#ffffff"),
        ["sassuolo"] = ("#00a752", "#000000"), ["cagliari"] = ("#002350", "#a71e22"), ["verona"] = ("#1b3c84", "#ffd700"), ["hellas verona"] = ("#1b3c84", "#ffd700"),
        ["lecce"] = ("#ffed00", "#d7172f"), ["parma"] = ("#ffd200", "#1b4f9c"), ["empoli"] = ("#1d5ba4", "#ffffff"), ["como"] = ("#003e7e", "#ffffff"), ["monza"] = ("#e2001a", "#ffffff"),
        ["paris saint germain"] = ("#004170", "#da291c"), ["paris sg"] = ("#004170", "#da291c"), ["psg"] = ("#004170", "#da291c"), ["marseille"] = ("#2faee0", "#ffffff"),
        ["olympique de marseille"] = ("#2faee0", "#ffffff"), ["lyon"] = ("#ffffff", "#1d3d8f"), ["olympique lyonnais"] = ("#ffffff", "#1d3d8f"), ["monaco"] = ("#e51b22", "#ffffff"), ["as monaco"] = ("#e51b22", "#ffffff"),
        ["lille"] = ("#e01e13", "#20325f"), ["nice"] = ("#c8102e", "#000000"), ["rennes"] = ("#e13327", "#000000"), ["lens"] = ("#ffe600", "#e30613"), ["rc lens"] = ("#ffe600", "#e30613"),
        ["nantes"] = ("#fcd405", "#00845a"), ["strasbourg"] = ("#009fe3", "#ffffff"), ["montpellier"] = ("#002e6d", "#f37021"), ["reims"] = ("#e30613", "#ffffff"),
        ["toulouse"] = ("#5b2a86", "#ffffff"), ["brest"] = ("#e30613", "#ffffff"), ["st etienne"] = ("#009a44", "#ffffff"), ["saint-etienne"] = ("#009a44", "#ffffff"), ["bordeaux"] = ("#001c50", "#ffffff"),
        ["ajax"] = ("#d2122e", "#ffffff"), ["psv"] = ("#ed1c24", "#ffffff"), ["psv eindhoven"] = ("#ed1c24", "#ffffff"), ["feyenoord"] = ("#ed1c24", "#ffffff"),
        ["az alkmaar"] = ("#db0021", "#ffffff"), ["az"] = ("#db0021", "#ffffff"), ["twente"] = ("#e30613", "#ffffff"), ["utrecht"] = ("#e30613", "#ffffff"),
        ["benfica"] = ("#e83030", "#ffffff"), ["porto"] = ("#00428c", "#ffffff"), ["sporting"] = ("#008057", "#ffffff"), ["sp lisbon"] = ("#008057", "#ffffff"),
        ["sporting cp"] = ("#008057", "#ffffff"), ["braga"] = ("#e30613", "#ffffff"), ["sp braga"] = ("#e30613", "#ffffff"), ["guimaraes"] = ("#ffffff", "#000000"),
        ["club brugge"] = ("#0068b3", "#000000"), ["club brugge kv"] = ("#0068b3", "#000000"), ["anderlecht"] = ("#5a2d82", "#ffffff"), ["genk"] = ("#0055a5", "#ffffff"),
        ["standard"] = ("#e30613", "#ffffff"), ["gent"] = ("#004f9f", "#ffffff"), ["antwerp"] = ("#e30613", "#ffffff"),
        ["celtic"] = ("#018749", "#ffffff"), ["rangers"] = ("#1b458f", "#ffffff"), ["aberdeen"] = ("#e2001a", "#ffffff"), ["hearts"] = ("#8e1b3a", "#ffffff"),
        ["hibernian"] = ("#00843d", "#ffffff"), ["dundee united"] = ("#f58025", "#000000"), ["dundee"] = ("#00205b", "#ffffff"), ["motherwell"] = ("#fbb800", "#8a1538"),
        ["kilmarnock"] = ("#004b93", "#ffffff"), ["st mirren"] = ("#000000", "#ffffff"), ["st johnstone"] = ("#0055a5", "#ffffff"), ["ross county"] = ("#002e6d", "#e30613"),
        ["galatasaray"] = ("#a90432", "#fdb912"), ["fenerbahce"] = ("#163962", "#ffed00"), ["besiktas"] = ("#000000", "#ffffff"), ["trabzonspor"] = ("#7e1c34", "#8bb9e7"),
        ["olympiakos"] = ("#e30613", "#ffffff"), ["olympiacos"] = ("#e30613", "#ffffff"), ["panathinaikos"] = ("#007a3d", "#ffffff"), ["aek"] = ("#ffde00", "#000000"), ["aek athens"] = ("#ffde00", "#000000"),
        ["paok"] = ("#000000", "#ffffff"),
    };

    public static string Key(string name)
    {
        var s = name.Normalize(NormalizationForm.FormD);
        var sb = new StringBuilder();
        foreach (var c in s)
        {
            if (System.Globalization.CharUnicodeInfo.GetUnicodeCategory(c) == System.Globalization.UnicodeCategory.NonSpacingMark) continue;
            sb.Append(char.IsLetterOrDigit(c) ? char.ToLowerInvariant(c) : ' ');
        }
        var words = sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries).Where(w => w is not ("fc" or "afc" or "cf" or "sc" or "ssc" or "ac" or "as" or "rc" or "sv" or "1" or "us")).ToList();
        return string.Join(' ', words);
    }

    public static (string Primary, string Secondary)? KnownColors(string name)
    {
        var lower = name.ToLowerInvariant().Replace(".", "").Trim();
        if (Colors.TryGetValue(lower, out var c)) return c;
        if (Colors.TryGetValue(Key(name), out c)) return c;
        return null;
    }

    public static string Slug(string name) => Key(name).Replace(' ', '-');

    private static readonly HashSet<string> ClubTypes = new(StringComparer.OrdinalIgnoreCase)
        { "afc", "fc", "cf", "sc", "ac", "as", "ss", "ssc", "rc", "rcd", "cd", "ud", "sd", "vfl", "vfb", "tsg", "sv", "fsv", "bsc", "ogc", "kv", "krc", "us", "sk", "fk", "sl", "cp", "spvgg", "and", "de", "of" };

    private static readonly HashSet<string> Suffixes = new(StringComparer.OrdinalIgnoreCase)
        { "united", "utd", "city", "town", "county", "rovers", "wanderers", "athletic", "albion", "hotspur", "calcio", "club" };

    private static readonly Dictionary<string, string> Codes = new(StringComparer.OrdinalIgnoreCase)
    {
        ["manchester united"] = "MUN", ["man united"] = "MUN", ["manchester city"] = "MCI", ["man city"] = "MCI",
        ["sheffield united"] = "SHU", ["sheffield utd"] = "SHU", ["sheffield wednesday"] = "SHW", ["sheffield weds"] = "SHW",
        ["bristol city"] = "BRC", ["bristol rovers"] = "BRR", ["crystal palace"] = "CRY", ["aston villa"] = "AVL", ["west ham"] = "WHU",
        ["west ham united"] = "WHU", ["bayern munich"] = "FCB", ["paris saint-germain"] = "PSG", ["paris sg"] = "PSG", ["inter"] = "INT",
    };

    public static string Code(string name)
    {
        if (Codes.TryGetValue(name.Trim(), out var known)) return known;
        var words = name.Replace("'", "").Replace(".", " ").Split([' ', '-'], StringSplitOptions.RemoveEmptyEntries)
            .Where(w => !ClubTypes.Contains(w) && !w.All(char.IsDigit)).ToList();
        if (words.Count == 0) return name.Length >= 3 ? name[..3].ToUpperInvariant() : name.ToUpperInvariant();
        if (words.Count >= 3) return string.Concat(words.Take(3).Select(w => char.ToUpperInvariant(w[0])));
        var main = words.Count == 2 && Suffixes.Contains(words[1]) ? words[0] : words.Count == 2 ? $"{words[0][0]}{words[1]}" : words[0];
        return (main.Length >= 3 ? main[..3] : main).ToUpperInvariant();
    }
}
