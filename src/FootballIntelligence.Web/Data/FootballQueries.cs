namespace FootballIntelligence.Web.Data;

public class FootballQueries(Database db)
{
    private static readonly string MatchSelect = $"""
        SELECT m.match_id, m.competition, c.name competition_name, m.season, m.match_date, m.kickoff, m.status,
               m.home_team_id, m.away_team_id, {Database.DisplayName("th.team_id", "th.name")} home_name, {Database.DisplayName("ta.team_id", "ta.name")} away_name,
               m.fthg, m.ftag, m.hthg, m.htag, m.hs, m.aws, m.hst, m.ast, m.hc, m.ac, m.hf, m.af, m.hy, m.ay, m.hr, m.ar, m.referee,
               p.p_home, p.p_draw, p.p_away, p.lambda_home, p.lambda_away, p.model_version, p.purpose,
               p.created_at prediction_created, p.data_cutoff, p.markets,
               (SELECT f.reason FROM prediction_flags f WHERE f.prediction_id=p.prediction_id LIMIT 1) prediction_flag
        FROM matches m
        JOIN competitions c ON c.code = m.competition
        JOIN teams th ON th.team_id = m.home_team_id
        JOIN teams ta ON ta.team_id = m.away_team_id
        LEFT JOIN predictions p ON p.prediction_id = (
            SELECT p2.prediction_id FROM predictions p2
            WHERE p2.match_id = m.match_id
            ORDER BY CASE p2.purpose WHEN 'forward' THEN 0 ELSE 1 END, CASE WHEN p2.model_version LIKE 'bookmaker%' THEN 1 ELSE 0 END, p2.prediction_id DESC LIMIT 1)
        """;

    private const string TeamFilter = """
        (@team IS NULL OR th.name LIKE '%'||@team||'%' OR ta.name LIKE '%'||@team||'%'
         OR EXISTS (SELECT 1 FROM team_aliases a WHERE a.team_id IN (m.home_team_id, m.away_team_id) AND a.alias LIKE '%'||@team||'%'))
        """;

    public List<MatchRow> Upcoming(int limit, string? competition = null, string? team = null) => db.Query<MatchRow>(
        MatchSelect + $" WHERE m.status='scheduled' AND m.match_date >= @today AND (@competition IS NULL OR m.competition=@competition) AND {TeamFilter} ORDER BY m.match_date, m.kickoff LIMIT @limit",
        new { limit, competition, team, today = DateTime.Now.ToString("yyyy-MM-dd") }).ToList();

    public List<MatchRow> Between(string from, string to, string? competition, string? team, int limit) => db.Query<MatchRow>(
        MatchSelect + $" WHERE m.match_date BETWEEN @from AND @to AND (@competition IS NULL OR m.competition=@competition) AND {TeamFilter} ORDER BY m.match_date, m.kickoff, m.competition LIMIT @limit",
        new { from, to, competition, team, limit }).ToList();

    public List<MatchRow> AwaitingResult(int limit) => db.Query<MatchRow>(
        MatchSelect + " WHERE m.status='scheduled' AND m.match_date < @today ORDER BY m.match_date DESC LIMIT @limit",
        new { limit, today = DateTime.Now.ToString("yyyy-MM-dd") }).ToList();

    public List<MatchRow> Recent(int limit, string? competition = null) => db.Query<MatchRow>(
        MatchSelect + " WHERE m.status='finished' AND (@competition IS NULL OR m.competition=@competition) ORDER BY m.match_date DESC, m.match_id DESC LIMIT @limit",
        new { limit, competition }).ToList();

    public List<MatchRow> OnDate(string date) => db.Query<MatchRow>(MatchSelect + " WHERE m.match_date=@date ORDER BY m.kickoff, m.competition", new { date }).ToList();

    public List<MatchRow> Search(string? competition, long? season, string? team, int limit, string? status = null) => db.Query<MatchRow>(
        MatchSelect + $"""
         WHERE (@competition IS NULL OR m.competition=@competition)
           AND (@status IS NULL OR m.status=@status)
           AND (@season IS NULL OR m.season=@season)
           AND {TeamFilter}
         ORDER BY m.match_date DESC, m.match_id DESC LIMIT @limit
        """, new { competition, season, team, limit, status }).ToList();

    public MatchRow? Match(long id) => db.Single<MatchRow>(MatchSelect + " WHERE m.match_id=@id", new { id });

    public List<MatchRow> PredictionsFor(long id) => db.Query<MatchRow>("""
        SELECT match_id, p_home, p_draw, p_away, lambda_home, lambda_away, model_version, purpose, created_at prediction_created, data_cutoff, markets
        FROM predictions WHERE match_id=@id ORDER BY prediction_id
        """, new { id }).ToList();

    public List<OddsRow> Odds(long id) => db.Query<OddsRow>("SELECT bookmaker, timing, home, draw, away, over25, under25 FROM odds WHERE match_id=@id ORDER BY bookmaker, timing", new { id }).ToList();

    public List<FormRow> Form(long teamId, string beforeDate, int limit) => db.Query<FormRow>($"""
        SELECT m.match_id, m.match_date, m.competition,
               CASE WHEN m.home_team_id=@teamId THEN {Database.DisplayName("ta.team_id", "ta.name")} ELSE {Database.DisplayName("th.team_id", "th.name")} END opponent,
               CASE WHEN m.home_team_id=@teamId THEN 'H' ELSE 'A' END venue,
               CASE WHEN m.home_team_id=@teamId THEN m.fthg ELSE m.ftag END goals_for,
               CASE WHEN m.home_team_id=@teamId THEN m.ftag ELSE m.fthg END goals_against,
               CASE WHEN m.home_team_id=@teamId THEN m.hs ELSE m.aws END shots_for,
               CASE WHEN m.home_team_id=@teamId THEN m.aws ELSE m.hs END shots_against
        FROM matches m JOIN teams th ON th.team_id=m.home_team_id JOIN teams ta ON ta.team_id=m.away_team_id
        WHERE (m.home_team_id=@teamId OR m.away_team_id=@teamId) AND m.status='finished' AND m.match_date < @beforeDate
        ORDER BY m.match_date DESC LIMIT @limit
        """, new { teamId, beforeDate, limit }).ToList();

    public List<MatchRow> HeadToHead(long homeId, long awayId, string beforeDate, int limit) => db.Query<MatchRow>(
        MatchSelect + """
         WHERE m.status='finished' AND m.match_date < @beforeDate
           AND ((m.home_team_id=@homeId AND m.away_team_id=@awayId) OR (m.home_team_id=@awayId AND m.away_team_id=@homeId))
         ORDER BY m.match_date DESC LIMIT @limit
        """, new { homeId, awayId, beforeDate, limit }).ToList();

    public StyleProfile? Style(long teamId, string beforeDate, int last = 10) => db.Single<StyleProfile>("""
        WITH recent AS (
            SELECT CASE WHEN home_team_id=@teamId THEN fthg ELSE ftag END gf, CASE WHEN home_team_id=@teamId THEN ftag ELSE fthg END ga,
                   CASE WHEN home_team_id=@teamId THEN hs ELSE aws END sf, CASE WHEN home_team_id=@teamId THEN aws ELSE hs END sa,
                   CASE WHEN home_team_id=@teamId THEN hst ELSE ast END tf, CASE WHEN home_team_id=@teamId THEN ast ELSE hst END ta,
                   CASE WHEN home_team_id=@teamId THEN hc ELSE ac END cf, CASE WHEN home_team_id=@teamId THEN ac ELSE hc END ca,
                   CASE WHEN home_team_id=@teamId THEN hf ELSE af END ff, CASE WHEN home_team_id=@teamId THEN hy ELSE ay END yc,
                   CASE WHEN home_team_id=@teamId THEN 1.0 ELSE 0.0 END is_home
            FROM matches WHERE (home_team_id=@teamId OR away_team_id=@teamId) AND status='finished' AND match_date < @beforeDate
            ORDER BY match_date DESC LIMIT @last)
        SELECT COUNT(*) matches, AVG(gf) goals_for, AVG(ga) goals_against, AVG(sf) shots_for, AVG(sa) shots_against, AVG(tf) sot_for, AVG(ta) sot_against,
               AVG(cf) corners_for, AVG(ca) corners_against, AVG(ff) fouls_for, AVG(yc) yellows, AVG(is_home) home_share
        FROM recent
        """, new { teamId, beforeDate, last });

    public long MatchesInPriorYear(long teamId, string date) => db.Single<long>(
        "SELECT COUNT(*) FROM matches WHERE (home_team_id=@teamId OR away_team_id=@teamId) AND status='finished' AND match_date < @date AND match_date >= date(@date, '-365 day')",
        new { teamId, date });

    public long? StatsBombMatchFor(long matchId) => db.Single<long?>("SELECT sb_match_id FROM sb_matches WHERE match_id=@matchId", new { matchId });

    public List<LineupRow> Lineup(long sbMatchId) => db.Query<LineupRow>("""
        SELECT a.player_id, p.name, a.side, a.position, a.started, a.minute_on, a.minute_off, a.minutes, a.goals, a.assists, a.xg, a.xa,
               a.shots, a.key_passes, a.progressive_passes, a.pressures, a.yellow, a.red
        FROM appearances a JOIN players p ON p.player_id=a.player_id
        WHERE a.sb_match_id=@sbMatchId ORDER BY a.side DESC, a.started DESC, a.minute_on
        """, new { sbMatchId }).ToList();

    public TeamInfo? Team(long id) => db.Single<TeamInfo>($"""
        SELECT t.team_id, {Database.DisplayName("t.team_id", "t.name")} name, t.country,
               (SELECT m.competition FROM matches m WHERE m.home_team_id=t.team_id OR m.away_team_id=t.team_id ORDER BY m.match_date DESC LIMIT 1) competition,
               (SELECT m.season FROM matches m WHERE (m.home_team_id=t.team_id OR m.away_team_id=t.team_id) AND m.status='finished' ORDER BY m.match_date DESC LIMIT 1) season,
               (SELECT MAX(m.match_date) FROM matches m WHERE (m.home_team_id=t.team_id OR m.away_team_id=t.team_id) AND m.status='finished') last_played
        FROM teams t WHERE t.team_id=@id
        """, new { id });

    public List<TeamInfo> FindTeams(string search, int limit) => db.Query<TeamInfo>($"""
        SELECT t.team_id, {Database.DisplayName("t.team_id", "t.name")} name, t.country,
               (SELECT m.competition FROM matches m WHERE m.home_team_id=t.team_id OR m.away_team_id=t.team_id ORDER BY m.match_date DESC LIMIT 1) competition,
               NULL season,
               (SELECT MAX(m.match_date) FROM matches m WHERE (m.home_team_id=t.team_id OR m.away_team_id=t.team_id) AND m.status='finished') last_played
        FROM teams t
        WHERE t.name LIKE '%'||@search||'%' OR EXISTS (SELECT 1 FROM team_aliases a WHERE a.team_id=t.team_id AND a.alias LIKE '%'||@search||'%')
        ORDER BY last_played DESC LIMIT @limit
        """, new { search, limit }).ToList();

    public List<MatchRow> TeamMatches(long teamId, string status, int limit) => db.Query<MatchRow>(
        MatchSelect + $" WHERE (m.home_team_id=@teamId OR m.away_team_id=@teamId) AND m.status=@status {(status == "scheduled" ? "AND m.match_date >= @today" : "")} ORDER BY m.match_date {(status == "finished" ? "DESC" : "ASC")}, m.kickoff LIMIT @limit",
        new { teamId, status, limit, today = DateTime.Now.ToString("yyyy-MM-dd") }).ToList();

    public List<StandingRow> Standings(string competition, long season) => db.Query<StandingRow>($"""
        WITH rows AS (
            SELECT home_team_id team_id, fthg gf, ftag ga, CASE WHEN fthg>ftag THEN 3 WHEN fthg=ftag THEN 1 ELSE 0 END pts, 1 home FROM matches
            WHERE competition=@competition AND season=@season AND status='finished'
            UNION ALL
            SELECT away_team_id, ftag, fthg, CASE WHEN ftag>fthg THEN 3 WHEN ftag=fthg THEN 1 ELSE 0 END, 0 FROM matches
            WHERE competition=@competition AND season=@season AND status='finished')
        SELECT r.team_id, {Database.DisplayName("r.team_id", "t.name")} name, COUNT(*) played, SUM(r.pts=3) won, SUM(r.pts=1) drawn, SUM(r.pts=0) lost,
               SUM(r.gf) goals_for, SUM(r.ga) goals_against, SUM(r.pts) points
        FROM rows r JOIN teams t ON t.team_id=r.team_id
        GROUP BY r.team_id ORDER BY points DESC, SUM(r.gf)-SUM(r.ga) DESC, SUM(r.gf) DESC, name
        """, new { competition, season }).ToList();

    public TrackRecord Record(string purpose) => db.Single<TrackRecord>("""
        WITH scored AS (
            SELECT p.p_home, p.p_draw, p.p_away, CASE WHEN m.fthg>m.ftag THEN 0 WHEN m.fthg=m.ftag THEN 1 ELSE 2 END y
            FROM predictions p JOIN matches m ON m.match_id=p.match_id
            WHERE p.purpose=@purpose AND m.status='finished' AND m.fthg IS NOT NULL
              AND p.prediction_id = (SELECT MAX(p2.prediction_id) FROM predictions p2 WHERE p2.match_id=p.match_id AND p2.purpose=@purpose
                                     AND p2.model_version NOT LIKE 'bookmaker%' AND p2.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags)))
        SELECT COUNT(*) n,
               COALESCE(SUM(CASE WHEN (y=0 AND p_home>=p_draw AND p_home>=p_away) OR (y=1 AND p_draw>p_home AND p_draw>=p_away) OR (y=2 AND p_away>p_home AND p_away>p_draw) THEN 1 ELSE 0 END), 0) top_pick_hits,
               AVG(-ln(CASE y WHEN 0 THEN p_home WHEN 1 THEN p_draw ELSE p_away END)) log_loss
        FROM scored
        """, new { purpose }) ?? new TrackRecord();

    private const string SettledFrom = """
        FROM predictions p JOIN matches m ON m.match_id=p.match_id
        JOIN teams th ON th.team_id=m.home_team_id JOIN teams ta ON ta.team_id=m.away_team_id
        WHERE m.status='finished' AND m.fthg IS NOT NULL AND p.purpose=@purpose
          AND p.prediction_id = (SELECT MAX(p2.prediction_id) FROM predictions p2 WHERE p2.match_id=p.match_id AND p2.purpose=@purpose
                                 AND p2.model_version NOT LIKE 'bookmaker%' AND p2.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags))
          AND (@competition IS NULL OR m.competition=@competition)
          AND (@team IS NULL OR th.name LIKE '%'||@team||'%' OR ta.name LIKE '%'||@team||'%'
               OR EXISTS (SELECT 1 FROM team_aliases a WHERE a.team_id IN (m.home_team_id, m.away_team_id) AND a.alias LIKE '%'||@team||'%'))
          AND (@hit IS NULL OR @hit = (CASE WHEN (m.fthg>m.ftag AND p.p_home>=p.p_draw AND p.p_home>=p.p_away)
                                             OR (m.fthg=m.ftag AND p.p_draw>p.p_home AND p.p_draw>=p.p_away)
                                             OR (m.fthg<m.ftag AND p.p_away>p.p_home AND p.p_away>p.p_draw) THEN 1 ELSE 0 END))
        """;

    public (long Total, long Hits, double? MeanActual, double? LogLoss) SettledSummary(string purpose, string? competition, string? team, int? hit)
    {
        var r = db.Single<(long, long, double?, double?)>($"""
            SELECT COUNT(*),
                   COALESCE(SUM(CASE WHEN (m.fthg>m.ftag AND p.p_home>=p.p_draw AND p.p_home>=p.p_away) OR (m.fthg=m.ftag AND p.p_draw>p.p_home AND p.p_draw>=p.p_away) OR (m.fthg<m.ftag AND p.p_away>p.p_home AND p.p_away>p.p_draw) THEN 1 ELSE 0 END), 0),
                   AVG(CASE WHEN m.fthg>m.ftag THEN p.p_home WHEN m.fthg=m.ftag THEN p.p_draw ELSE p.p_away END),
                   AVG(-ln(CASE WHEN m.fthg>m.ftag THEN p.p_home WHEN m.fthg=m.ftag THEN p.p_draw ELSE p.p_away END))
            {SettledFrom}
            """, new { purpose, competition, team, hit });
        return r;
    }

    public List<MatchRow> Settled(string purpose, string? competition, string? team, int? hit, int limit, int offset) => db.Query<MatchRow>($"""
        SELECT m.match_id, m.competition, m.season, m.match_date, m.kickoff, m.status, m.home_team_id, m.away_team_id,
               {Database.DisplayName("th.team_id", "th.name")} home_name, {Database.DisplayName("ta.team_id", "ta.name")} away_name, m.fthg, m.ftag,
               p.p_home, p.p_draw, p.p_away, p.lambda_home, p.lambda_away, p.model_version, p.purpose, p.created_at prediction_created, p.data_cutoff, p.markets
        {SettledFrom}
        ORDER BY m.match_date DESC, m.match_id DESC LIMIT @limit OFFSET @offset
        """, new { purpose, competition, team, hit, limit, offset }).ToList();

    public List<string> Competitions() => db.Query<string>("SELECT code FROM competitions ORDER BY tier, code").ToList();

    public List<PlayerSummary> Players(string? search, string? team, string? position, int limit) => db.Query<PlayerSummary>("""
        SELECT a.player_id, p.name, MAX(a.team_name) team, (SELECT a2.position FROM appearances a2 WHERE a2.player_id=a.player_id AND a2.position IS NOT NULL GROUP BY a2.position ORDER BY COUNT(*) DESC LIMIT 1) position, COUNT(*) apps, SUM(a.started) starts, SUM(a.minutes) minutes,
               SUM(a.goals) goals, SUM(a.assists) assists, SUM(a.xg) xg, SUM(a.xa) xa, SUM(a.shots) shots, SUM(a.key_passes) key_passes,
               SUM(a.progressive_passes) progressive_passes, SUM(a.progressive_carries) progressive_carries, SUM(a.pressures) pressures,
               SUM(a.tackles) tackles, SUM(a.interceptions) interceptions, SUM(a.recoveries) recoveries, SUM(a.yellow) yellow, SUM(a.red) red,
               SUM(a.gk_shots_faced) gk_shots_faced, SUM(a.gk_goals_conceded) gk_goals_conceded, SUM(a.gk_xg_faced) gk_xg_faced
        FROM appearances a JOIN players p ON p.player_id=a.player_id
        WHERE (@search IS NULL OR p.name LIKE '%'||@search||'%') AND (@team IS NULL OR a.team_name=@team)
        GROUP BY a.player_id
        HAVING (@position IS NULL OR position LIKE '%'||@position||'%') AND SUM(a.minutes) >= 450
        ORDER BY SUM(a.xg)+SUM(a.xa) DESC LIMIT @limit
        """, new { search, team, position, limit }).ToList();

    private Tuple<long, List<TeamTactics>>? tacticsCache;

    public List<TeamTactics> TeamTactics()
    {
        var processed = db.Single<long>("SELECT COUNT(*) FROM sb_matches WHERE processed=1");
        if (tacticsCache is { } c && c.Item1 == processed) return c.Item2;
        var rows = LoadTeamTactics();
        tacticsCache = Tuple.Create(processed, rows);
        return rows;
    }

    private List<TeamTactics> LoadTeamTactics() => db.Query<TeamTactics>("""
        WITH sides AS (
            SELECT sb_match_id, 'home' side, home_name team, competition FROM sb_matches WHERE processed=1
            UNION ALL SELECT sb_match_id, 'away', away_name, competition FROM sb_matches WHERE processed=1),
        shots AS (
            SELECT s.team, s.competition, s.sb_match_id,
                   SUM(CASE WHEN e.side=s.side THEN e.xg END) xg_for, SUM(CASE WHEN e.side<>s.side THEN e.xg END) xg_against,
                   SUM(e.side=s.side) shots_for, SUM(e.side<>s.side) shots_against,
                   SUM(CASE WHEN e.side=s.side AND e.detail<>'Penalty' AND e.pattern IN ('From Corner','From Free Kick') THEN e.xg ELSE 0.0 END) sp_for,
                   SUM(CASE WHEN e.side<>s.side AND e.detail<>'Penalty' AND e.pattern IN ('From Corner','From Free Kick') THEN e.xg ELSE 0.0 END) sp_against,
                   SUM(CASE WHEN e.side=s.side AND e.pattern='From Counter' THEN e.xg ELSE 0.0 END) counter_for,
                   SUM(CASE WHEN e.side<>s.side AND e.pattern='From Counter' THEN e.xg ELSE 0.0 END) counter_against,
                   SUM(CASE WHEN e.side=s.side AND e.detail='Penalty' THEN e.xg ELSE 0.0 END) pen_for
            FROM sides s JOIN sb_events e ON e.sb_match_id=s.sb_match_id AND e.kind='shot'
            GROUP BY s.team, s.competition, s.sb_match_id),
        mins AS (
            SELECT s.team, s.sb_match_id,
                   SUM(CASE WHEN m.side=s.side THEN m.possession_seconds END) poss_own, SUM(m.possession_seconds) poss_all,
                   SUM(CASE WHEN m.side=s.side THEN m.def_actions_high END) def_high, SUM(CASE WHEN m.side<>s.side THEN m.passes_own60 END) opp_own60,
                   SUM(CASE WHEN m.side<>s.side THEN m.def_actions_high END) opp_def_high, SUM(CASE WHEN m.side=s.side THEN m.passes_own60 END) own60,
                   SUM(CASE WHEN m.side=s.side THEN m.final_third_passes END) f3_own, SUM(m.final_third_passes) f3_all,
                   SUM(CASE WHEN m.side=s.side THEN m.pressures END) pressures, SUM(CASE WHEN m.side=s.side THEN m.box_touches END) box
            FROM sides s JOIN sb_team_minutes m ON m.sb_match_id=s.sb_match_id
            GROUP BY s.team, s.sb_match_id),
        goals AS (
            SELECT s.team, s.sb_match_id, SUM(e.side=s.side) gf, SUM(e.side<>s.side) ga
            FROM sides s LEFT JOIN sb_events e ON e.sb_match_id=s.sb_match_id AND e.kind='goal'
            GROUP BY s.team, s.sb_match_id)
        SELECT sh.team, sh.competition, COUNT(*) matches,
               CAST(SUM(sh.xg_for) AS REAL) xg_for, CAST(SUM(sh.xg_against) AS REAL) xg_against, CAST(SUM(sh.shots_for) AS REAL) shots_for, CAST(SUM(sh.shots_against) AS REAL) shots_against,
               CAST(SUM(sh.sp_for) AS REAL) set_piece_xg_for, CAST(SUM(sh.sp_against) AS REAL) set_piece_xg_against, CAST(SUM(sh.counter_for) AS REAL) counter_xg_for, CAST(SUM(sh.counter_against) AS REAL) counter_xg_against,
               CAST(SUM(sh.pen_for) AS REAL) penalty_xg_for,
               CAST(SUM(mi.poss_own) AS REAL) / SUM(mi.poss_all) possession,
               CAST(SUM(mi.opp_own60) AS REAL) / NULLIF(SUM(mi.def_high), 0) ppda,
               CAST(SUM(mi.own60) AS REAL) / NULLIF(SUM(mi.opp_def_high), 0) opp_ppda,
               CAST(SUM(mi.f3_own) AS REAL) / NULLIF(SUM(mi.f3_all), 0) field_tilt,
               CAST(SUM(mi.pressures) AS REAL) / COUNT(*) pressures_per_match, CAST(SUM(mi.box) AS REAL) / COUNT(*) box_touches_per_match,
               CAST(SUM(g.gf) AS REAL) goals_for, CAST(SUM(g.ga) AS REAL) goals_against
        FROM shots sh JOIN mins mi ON mi.team=sh.team AND mi.sb_match_id=sh.sb_match_id JOIN goals g ON g.team=sh.team AND g.sb_match_id=sh.sb_match_id
        GROUP BY sh.team, sh.competition HAVING COUNT(*) >= 10
        """).ToList();

    public List<string> PlayerTeams() => db.Query<string>("SELECT DISTINCT team_name FROM appearances ORDER BY team_name").ToList();

    public PlayerSummary? Player(long id) => db.Single<PlayerSummary>("""
        SELECT a.player_id, p.name, MAX(a.team_name) team, (SELECT a2.position FROM appearances a2 WHERE a2.player_id=a.player_id AND a2.position IS NOT NULL GROUP BY a2.position ORDER BY COUNT(*) DESC LIMIT 1) position, COUNT(*) apps, SUM(a.started) starts, SUM(a.minutes) minutes,
               SUM(a.goals) goals, SUM(a.assists) assists, SUM(a.xg) xg, SUM(a.xa) xa, SUM(a.shots) shots, SUM(a.key_passes) key_passes,
               SUM(a.progressive_passes) progressive_passes, SUM(a.progressive_carries) progressive_carries, SUM(a.pressures) pressures,
               SUM(a.tackles) tackles, SUM(a.interceptions) interceptions, SUM(a.recoveries) recoveries, SUM(a.yellow) yellow, SUM(a.red) red,
               SUM(a.gk_shots_faced) gk_shots_faced, SUM(a.gk_goals_conceded) gk_goals_conceded, SUM(a.gk_xg_faced) gk_xg_faced
        FROM appearances a JOIN players p ON p.player_id=a.player_id WHERE a.player_id=@id GROUP BY a.player_id
        """, new { id });

    public List<PlayerMatchRow> PlayerMatches(long id) => db.Query<PlayerMatchRow>("""
        SELECT a.sb_match_id, s.match_date, a.team_name,
               CASE WHEN a.side='home' THEN s.away_name ELSE s.home_name END opponent,
               a.position, a.started, a.minutes, a.goals, a.assists, a.xg, a.xa, a.shots, a.key_passes, a.progressive_passes, a.pressures,
               COALESCE(CAST(julianday(s.match_date) - julianday(LAG(s.match_date) OVER (ORDER BY s.match_date)) AS INTEGER), 0) days_since_previous
        FROM appearances a JOIN sb_matches s ON s.sb_match_id=a.sb_match_id WHERE a.player_id=@id ORDER BY s.match_date
        """, new { id }).ToList();
}
