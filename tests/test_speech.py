import time

from bookworm.speech import BackgroundSpeaker


class SlowSay:
    def __init__(self):
        self.spoken = []

    def __call__(self, text):
        self.spoken.append(text)
        time.sleep(0.2)

    def stop(self):
        pass


def test_stop_then_resume_continues_from_the_interrupted_sentence():
    say = SlowSay()
    s = BackgroundSpeaker(say)
    s.say("One. Two. Three.")
    time.sleep(0.3)                     # cut off during "Two."
    s.stop()
    s.wait()
    assert say.spoken == ["One.", "Two."]
    assert s.resume()
    s.wait()
    assert say.spoken[2:] == ["Two.", "Three."]
    assert not s.resume()               # nothing left to continue


def test_short_replies_are_not_kept_for_resume():
    say = SlowSay()
    s = BackgroundSpeaker(say)
    s.say("Okay, reading.", resumable=False)
    s.say("Page one. Page two.")
    time.sleep(0.3)                     # cut off during "Page one."
    s.stop()
    s.say("Sorry.", resumable=False)    # a reply after the stop must not replace the saved place
    s.wait()
    s.stop()                            # a second stop with nothing playing keeps it too
    assert s.resume()
    s.wait()
    assert say.spoken[-2:] == ["Page one.", "Page two."]
