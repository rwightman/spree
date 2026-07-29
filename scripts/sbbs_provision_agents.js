/*
 * Provision Synchronet users from a resolved agent registry JSON file.
 *
 * Expected input:
 * {
 *   "agents": [
 *     {
 *       "agent_id": "gemma-local-001",
 *       "bbs_alias": "GemmaOne",
 *       "bbs_password": "secret",
 *       "security_level": 50
 *     }
 *   ]
 * }
 */

"use strict";

function fail(message) {
    writeln("ERROR: " + message);
    exit(1);
}

function readJson(path) {
    var file = new File(path);
    if (!file.open("r")) fail("cannot open " + path);
    var text = file.read();
    file.close();
    return JSON.parse(text);
}

function provisionAgent(agent) {
    var alias = agent.bbs_alias;
    var password = agent.bbs_password;
    if (typeof alias !== "string" || alias.length < 1) fail("missing bbs_alias");
    if (typeof password !== "string" || password.length < 1) fail("missing bbs_password for " + alias);

    var userNumber = system.matchuser(alias);
    var created = false;
    var usr;
    if (userNumber > 0) {
        usr = new User(userNumber);
    } else {
        usr = system.new_user(alias);
        created = true;
    }

    usr.alias = alias;
    usr.handle = alias;
    usr.name = alias;
    usr.security.password = password;
    if (typeof agent.security_level === "number") usr.security.level = agent.security_level;

    // system.new_user() intentionally leaves profile fields blank. The stock
    // Synchronet logon module prompts for those fields before the first session,
    // which is appropriate for people but would consume an agent's game budget.
    // Fill non-personal bot defaults without replacing values a sysop set.
    if (!usr.location) usr.location = "Spree";
    if (!usr.gender) usr.gender = "X";
    if (!usr.birthdate) usr.birthdate = "01/01/2000";
    if (!usr.netmail) usr.netmail = alias.toLowerCase() + "@spree.invalid";

    writeln((created ? "created" : "updated") + " " + agent.agent_id + " -> " + alias);
}

if (argv.length < 1) fail("usage: jsexec sbbs_provision_agents.js <agents.json>");

var config = readJson(argv[0]);
if (!(config.agents instanceof Array)) fail("agents must be an array");

for (var i = 0; i < config.agents.length; i++) {
    provisionAgent(config.agents[i]);
}
