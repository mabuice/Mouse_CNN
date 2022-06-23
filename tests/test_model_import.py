import mousenet

def test_generic_import():
    model = mousenet.load(architecture="default", pretraining=None)

def test_retinotopic_import():
    model = mousenet.load(architecture="retinotopic", pretraining=None)

def test_imagenet_import():
    model = mousenet.load(architecture="default", pretraining="imagenet")

def test_kaiming_import():
    model = mousenet.load(architecture="default", pretraining="kaiming")
