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

    public string? Kv(string key) => Single<string>("SELECT value FROM kv WHERE key=@key", new { key });
}
