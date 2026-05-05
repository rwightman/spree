/*
 * Grant turns to one Synchronet TW2 player without resetting the world.
 *
 * Usage inside the container:
 *   jsexec /tmp/sbbs_tw2_grant_turns.js RLoginSmoke 30
 *
 * If turns is omitted, the script uses Settings.TurnsPerDay.
 */

"use strict";

var startup_path = "/sbbs-data/xtrn/tw2/";

load(startup_path + "filename.js");
load("json-client.js");

var LOCK_WRITE = 2;
var LOCK_READ = 1;
var db;

load(fname("gamesettings.js"));

var Settings = new GameSettings();
if (db === undefined) {
    writeln("ERROR: unable to connect to TW2 JSON service");
    exit(1);
}

var alias = argv[0];
if (!alias) {
    writeln("ERROR: player alias is required");
    exit(2);
}

var turns = argv[1] === undefined ? Settings.TurnsPerDay : parseInt(argv[1], 10);
if (isNaN(turns) || turns < 0) {
    writeln("ERROR: turns must be a non-negative integer");
    exit(2);
}

db.lock(Settings.DB, "players", LOCK_WRITE);
var players = db.read(Settings.DB, "players");
var found = false;

for (var i = 1; i < players.length; i++) {
    var player = players[i];
    if (!player || player.Alias !== alias) {
        continue;
    }
    var oldTurns = player.TurnsLeft;
    player.TurnsLeft = turns;
    player.Online = false;
    players[i] = player;
    db.write(Settings.DB, "players", players);
    writeln(
        "TW2 turns updated: "
            + alias
            + " record="
            + i
            + " "
            + oldTurns
            + " -> "
            + turns
            + " sector="
            + player.Sector
    );
    found = true;
    break;
}

db.unlock(Settings.DB, "players");

if (!found) {
    writeln("ERROR: player not found: " + alias);
    exit(3);
}
