from abc import ABC


class ImagePathToInformationMapping(ABC):
    def __init__(self):
        pass

    def __call__(self, full_path):
        pass


class ImageNetInfoMapping(ImagePathToInformationMapping):
    """
        For ImageNet-like directory structures without sessions/conditions:
        .../{category}/{img_name}
    """

    def __call__(self, full_path):
        session_name = "session-1"
        img_name = full_path.split("/")[-1]
        condition = "NaN"
        category = full_path.split("/")[-2]

        return session_name, img_name, condition, category


class ImageNetCInfoMapping(ImagePathToInformationMapping):
    """
        For the ImageNet-C Dataset with path structure:
        ...{corruption function}/{corruption severity}/{category}/{img_name}
    """

    def __call__(self, full_path):
        session_name = "session-1"
        parts = full_path.split("/")
        img_name = parts[-1]
        category = parts[-2]
        severity = parts[-3]
        corruption = parts[-4]
        condition = "{}-{}".format(corruption, severity)
        return session_name, img_name, condition, category


class InfoMappingWithSessions(ImagePathToInformationMapping):
    """
        Directory/filename structure:
        .../{session_name}/{something}_{something}_{something}_{condition}_{category}_{img_name}
    """

    def __call__(self, full_path):
        session_name = full_path.split("/")[-2]
        img_name = full_path.split("/")[-1]
        condition = img_name.split("_")[3]
        category = img_name.split("_")[4]

        return session_name, img_name, condition, category
