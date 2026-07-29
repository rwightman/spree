@lredir -f E: linux\fs/sbbs-data/xtrn >NUL
@lredir -f F: linux\fs/sbbs/ctrl >NUL
@lredir -f G: linux\fs/sbbs/data >NUL
@lredir -f H: linux\fs/sbbs/exec >NUL

@SET TZ=UTC0

E:

IF "%STARTDIR%"=="" D:
IF NOT "%STARTDIR%"=="" CD %STARTDIR%

$CMDLINE

IF NOT "%1" == "TEST" exitemu
