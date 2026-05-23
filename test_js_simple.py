import socket
s = socket.socket()
s.settimeout(10)
s.connect(('127.0.0.1', 9999))
code = 'return (function(){try{return "OK"}catch(e){return "ERR"}})()'
msg = 'js ' + code
print('Sending:', repr(msg[:100]))
s.send(msg.encode())
r = s.recv(65536).decode()
s.close()
print('Result:', repr(r))
