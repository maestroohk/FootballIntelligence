using FootballIntelligence.Web.Components;
using FootballIntelligence.Web.Data;
using FootballIntelligence.Web.Services;

var builder = WebApplication.CreateBuilder(args);
builder.WebHost.UseStaticWebAssets();

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.Configure<EngineOptions>(builder.Configuration.GetSection("Engine"));
builder.Services.AddSingleton<Database>();
builder.Services.AddSingleton<FootballQueries>();
builder.Services.AddSingleton<ResearchQueries>();
builder.Services.AddSingleton<OperationsQueries>();
builder.Services.AddScoped<ViewMode>();
builder.Services.AddSingleton<ChangeFeed>();
builder.Services.AddHostedService(sp => sp.GetRequiredService<ChangeFeed>());
builder.Services.AddSingleton<EngineSupervisor>();
builder.Services.AddHostedService(sp => sp.GetRequiredService<EngineSupervisor>());

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
}
app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);

app.UseStaticFiles();
app.UseRouting();
app.UseAntiforgery();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.MapGet("/api/health", (OperationsQueries ops) => Results.Ok(ops.WorkerStatus()));
app.MapGet("/api/predictions/upcoming", (FootballQueries q) => Results.Ok(q.Upcoming(200)));

app.Run();
