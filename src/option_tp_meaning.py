def option_to_meaning(data):
    text = ""
    text += "• เริ่มเขียนจากพยัญชนะนำตำแหน่งที่ 1: ให้เขียนพยัญชนะตัวแรกของชื่อให้มีขนาดใหญ่และชัดเจน เพื่อสะท้อนถึงความคิดสร้างสรรค์และความมั่นใจ ควรหลีกเลี่ยงการเขียนตัวอักษรที่ไม่สมบูรณ์หรือมีการเติมหรือลดลงที่ไม่จำเป็น\n"
    text += "•"
    if(data.boss == 'round'):
        text += "•"
    elif(data.boss == 'edge'):
        text += "•"
    if(data.tilt == 'none'):
        text += "•"
    elif(data.tilt == 'tilt_up'):
        text += "•"
    if(data.symbol == 'none'):
        text += "•"
    elif(data.symbol == 'omega'):
        text += "•"
    elif(data.symbol == 'omega'):
        text += "•"
    if(data.point == 'checked_point'):
        text += "•"
    if(data.line == 'checked_line'):
        text += "•"
    if(data.circle == 'checked_circle'):
        text += "•"
    return text