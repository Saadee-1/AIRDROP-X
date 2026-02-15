' Double-click se AIRDROP-X sirf ek alag window mein open (bina console)
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = folder
shell.Run "pythonw main.py", 0, False
