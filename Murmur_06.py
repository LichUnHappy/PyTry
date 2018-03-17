# 粒度控制

# def SendContent(ServerAdr,PagePath,StartLine,EndLine,sender
#                 receiver,smtpsever,username,passpord):
#     http = httplib.HTTP(ServerAdr)
#     http.putrequest("Get", PagePath)
#     http.putheader("Accept", "text/html")
#     http.putheader("Accept", "text/plain")
#     http.endheaders()

#     httpcpde, httpmsg, headers = http.getreply()

#     if httpcode != 200: raise "Could not get document: Check URL and Path."
#     doc = http.getfile()
#     data = doc.read()
#     doc.close()
    
#     lstr = data.splitlines()
#     j = 0
#     for i in lstr:
#         j = j + 1
#         if i.strip() == StartLine: slice_start = j      # find slice start
#         elif i.strip() == EndLine: slice_end = j        # find slice end
    
#     subject = "Contented get from the web"
#     msg = MIMEText(string.join(lstr[slice_start:slice_end]),'plain','utf-8')
#     msg['subject'] = Header(subject, 'utf-8')
#     smtp = smtplib.SMTP()
#     smtp.connect(smtpserver)
#     smtp.login(username,password)
#     smtp.sendmail(sender, receiver, msg.as_string())
#     smtp.quit()


def GetContent(ServerAdr,PagePath):
    http = httplib.HTTP(ServerAdr)
    http.putrequest("Get", PagePath)
    http.putheader("Accept", "text/html")
    http.putheader("Accept", "text/plain")
    http.endheaders()

    httpcpde, httpmsg, headers = http.getreply()

    if httpcode != 200: raise "Could not get document: Check URL and Path."
    doc = http.getfile()
    data = doc.read()
    doc.close()
    return data


def ExtractData(inputstring, start_line, end_line):
    lstr = inputstring.splitlines()
    j = 0
    for i in lstr:
        j = j + 1
        if i.strip() == start_line: slice_start = j      # find slice start
        elif i.strip() == end_line: slice_end = j        # find slice end
    return lstr[slice_start:slice_end]
    
def SendEmail(sender,receiver,smtpserver,username,password,content):
    subject = "Contented get from the web"
    msg = MIMEText(content,'plain','utf-8')
    msg['subject'] = Header(subject, 'utf-8')
    smtp = smtplib.SMTP()
    smtp.connect(smtpserver)
    smtp.login(username,password)
    smtp.sendmail(sender, receiver, msg.as_string())
    smtp.quit()