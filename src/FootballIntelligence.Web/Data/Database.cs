using Dapper;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Options;

namespace FootballIntelligence.Web.Data;

public class EngineOptions
{
    public string DatabasePath { get; set; } = "../../data/football.db";
    public string PythonPath { get; set; } = "../../.venv/Scripts/python.exe";
    public string EngineDirectory { get; set; } = "../../engine";
    public bool StartWorker { get; set; } = true;
    public string Profile { get; set; } = Environment.GetEnvironmentVariable("FIE_PROFILE") ?? "full";
    public bool Light => string.Equals(Profile, "light", StringComparison.OrdinalIgnoreCase);
}

public class Database
{
    private readonly string connectionString;

    public Database(IOptions<EngineOptions> options, IWebHostEnvironment env)
    {
        Options = options.Value;
        DatabaseFile = Resolve(env, Options.DatabasePath);
        PythonFile = Resolve(env, Options.PythonPath);
        EngineDirectory = Resolve(env, Options.EngineDirectory);
        connectionString = new SqliteConnectionStringBuilder { DataSource = DatabaseFile, DefaultTimeout = 30 }.ToString();
        DefaultTypeMap.MatchNamesWithUnderscores = true;
    }

    public EngineOptions Options { get; }
    public string DatabaseFile { get; }
    public string PythonFile { get; }
    public string EngineDirectory { get; }
    public bool Exists => File.Exists(DatabaseFile);

    private static string Resolve(IWebHostEnvironment env, string path) =>
        Path.GetFullPath(Path.IsPathRooted(path) ? path : Path.Combine(env.ContentRootPath, path));

    public SqliteConnection Open()
    {
        var con = new SqliteConnection(connectionString);
        con.Open();
        con.Execute("PRAGMA busy_timeout=30000");
        return con;
    }

    public IEnumerable<T> Query<T>(string sql, object? args = null)
    {
        using var con = Open();
        return con.Query<T>(sql, args).ToList();
    }

    public T? Single<T>(string sql, object? args = null)
    {
        using var con = Open();
        return con.QueryFirstOrDefault<T>(sql, args);
    }

    public int Execute(string sql, object? args = null)
    {
        using var con = Open();
        return con.Execute(sql, args);
    }

    public static string DisplayName(string idColumn, string nameColumn) =>
        $"COALESCE((SELECT CASE WHEN a.alias LIKE '% FC' THEN substr(a.alias, 1, length(a.alias) - 3) WHEN a.alias LIKE 'FC %' THEN substr(a.alias, 4) WHEN a.alias LIKE 'AFC %' THEN substr(a.alias, 5) ELSE a.alias END " +
        $"FROM team_aliases a WHERE a.team_id = {idColumn} AND a.source IN ('statsbomb', 'openfootball') ORDER BY a.source = 'statsbomb' DESC LIMIT 1), {nameColumn})";

    public string? Kv(string key) => Single<string>("SELECT value FROM kv WHERE key=@key", new { key });
}
