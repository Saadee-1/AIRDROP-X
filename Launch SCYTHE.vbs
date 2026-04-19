' Double-click = AIRDROP-X in desktop window only (no console)
' Runs the .bat so the same "python" and PATH are used as when you run the bat manually
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = fso.BuildPath(folder, "launch_airdrop_x.bat")
' Run batch with hidden console (0), don't wait (False)
shell.Run "cmd.exe /c """ & batPath & """", 0, False
