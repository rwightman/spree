"use strict";

var lockPath = js.exec_dir + "inuse.sr";
if (file_exists(lockPath) && !file_remove(lockPath)) {
    log(LOG_ERR, "Unable to remove stale SRE lock: " + lockPath);
}
