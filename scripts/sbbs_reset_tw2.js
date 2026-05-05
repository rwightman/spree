/*
 * Non-interactive Trade Wars 2 reset/initialization for the local Synchronet
 * container. This mirrors the reset block in xtrn/tw2/twint500.js without
 * entering the UIFC configuration UI.
 */

"use strict";

var startup_path = "/sbbs-data/xtrn/tw2/";

load(startup_path + "filename.js");
load("json-client.js");

var LOCK_WRITE = 2;
var LOCK_READ = 1;
var db;

load(fname("gamesettings.js"));
load(fname("sector_map.js"));
load(fname("ports_map.js"));

var Settings = new GameSettings();
if (db === undefined) {
    writeln("ERROR: unable to connect to TW2 JSON service");
    exit(1);
}

Settings.save();

load(fname("ports.js"));
load(fname("planets.js"));
load(fname("teams.js"));
load(fname("sectors.js"));
load(fname("maint.js"));
load(fname("players.js"));
load(fname("messages.js"));
load(fname("computer.js"));
load(fname("input.js"));

function clone(value) {
    return eval(value.toSource());
}

function defaultsFrom(properties) {
    var out = {};
    for (var i = 0; i < properties.length; i++) {
        out[properties[i].prop] = clone(properties[i].def);
    }
    return out;
}

function resetAllPlayers() {
    var player = defaultsFrom(PlayerProperties);
    player.UserNumber = 0;
    player.Sector = 0;

    db.lock(Settings.DB, "players", LOCK_WRITE);
    db.write(Settings.DB, "players", []);
    db.push(Settings.DB, "players", { Excuse: "zero-based array padding" });
    for (var i = 0; i < Settings.MaxPlayers; i++) {
        db.push(Settings.DB, "players", clone(player));
    }
    db.unlock(Settings.DB, "players");
}

function resetAllPlanets() {
    db.lock(Settings.DB, "planets", LOCK_WRITE);
    db.write(Settings.DB, "planets", []);
    db.push(Settings.DB, "planets", clone(DefaultPlanet));
    for (var i = 0; i < Settings.MaxPlanets; i++) {
        db.push(Settings.DB, "planets", clone(DefaultPlanet));
    }
    db.unlock(Settings.DB, "planets");
}

function resetAllMessages() {
    db.write(Settings.DB, "log", [], LOCK_WRITE);
    db.push(
        Settings.DB,
        "log",
        { Date: strftime("%a %b %d %H:%M:%S %Z"), Message: " TW 500 initialized" },
        LOCK_WRITE
    );
    db.write(Settings.DB, "updates", [], LOCK_WRITE);
    db.write(Settings.DB, "radio", [], LOCK_WRITE);
}

function initializeTeams() {
    db.write(Settings.DB, "teams", [], LOCK_WRITE);
    db.push(Settings.DB, "teams", { Excuse: "zero-based array padding" }, LOCK_WRITE);
}

function initializeSectors() {
    db.lock(Settings.DB, "sectors", LOCK_WRITE);
    db.write(Settings.DB, "sectors", []);
    db.push(Settings.DB, "sectors", { Excuse: "zero-based array padding" });
    for (var i = 0; i < sector_map.length; i++) {
        var sector = clone(DefaultSector);
        for (var prop in sector_map[i]) {
            sector[prop] = clone(sector_map[i][prop]);
        }
        db.push(Settings.DB, "sectors", sector);
    }
    db.unlock(Settings.DB, "sectors");
}

function initializePorts() {
    db.lock(Settings.DB, "ports", LOCK_WRITE);
    db.write(Settings.DB, "ports", []);
    db.push(Settings.DB, "ports", clone(DefaultPort));

    for (var i = 0; i < ports_init.length; i++) {
        var port = clone(DefaultPort);
        for (var prop in ports_init[i]) {
            if (port[prop] !== undefined) {
                port[prop] = clone(ports_init[i][prop]);
            }
        }
        port.Production = [ports_init[i].OreProduction, ports_init[i].OrgProduction, ports_init[i].EquProduction];
        port.PriceVariance = [ports_init[i].OreDeduction, ports_init[i].OrgDeduction, ports_init[i].EquDeduction];

        db.lock(Settings.DB, "sectors." + ports_init[i].Sector, LOCK_WRITE);
        var sector = db.read(Settings.DB, "sectors." + ports_init[i].Sector);
        sector.Port = i + 1;
        db.write(Settings.DB, "sectors." + ports_init[i].Sector, sector);
        db.unlock(Settings.DB, "sectors." + ports_init[i].Sector);

        db.push(Settings.DB, "ports", port);
    }
    db.unlock(Settings.DB, "ports");
}

function initializeCabal() {
    var sector = db.read(Settings.DB, "sectors.85", LOCK_READ);
    sector_map[85].Fighters = 3000;
    sector_map[85].FightersOwner = -1;
    sector.Fighters = 3000;
    sector.FightersOwner = -1;
    db.write(Settings.DB, "sectors.85", sector, LOCK_WRITE);

    db.lock(Settings.DB, "cabals", LOCK_WRITE);
    db.write(Settings.DB, "cabals", []);
    db.push(Settings.DB, "cabals", clone(DefaultCabal));
    for (var i = 1; i < 10; i++) {
        var cabal = clone(DefaultCabal);
        if (i === 1) {
            cabal.Size = 3000;
            cabal.Sector = 85;
            cabal.Goal = 85;
        }
        db.write(Settings.DB, "cabals." + i, cabal);
    }
    db.unlock(Settings.DB, "cabals");
}

resetAllPlayers();
resetAllPlanets();
resetAllMessages();
initializeTeams();
initializeSectors();
initializePorts();
initializeCabal();
db.write(Settings.DB, "twopeng", [], LOCK_WRITE);

writeln("TW2 reset complete");
