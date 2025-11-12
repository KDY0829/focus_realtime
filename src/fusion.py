# src/fusion.py
def concentration_score(gaze, neck, emotion, blink,
                        w_gaze=0.4, w_neck=0.3, w_emotion=0.2, w_blink=0.1):
    vals = [gaze or 0.0, neck or 0.0, emotion or 0.0, blink or 0.0]
    ws   = [w_gaze, w_neck, w_emotion, w_blink]
    s = sum(a*b for a,b in zip(vals, ws))
    return round(max(0.0, min(1.0, s)), 4)

