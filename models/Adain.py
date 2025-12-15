def adaptive_instance_normalization(content_feat, style_feat):
    """
    by removing the assertion, tensor shape should be checked manually.
    """
    
    size = content_feat.size()

    style_mean, style_std = style_feat.mean(dim=[dim for dim in range(2, len(style_feat.shape))]), style_feat.std(dim=[dim for dim in range(2, len(style_feat.shape))])

    while len(style_mean.size()) < len(size):
        style_mean = style_mean.unsqueeze(dim=-1)
    while len(style_std.size()) < len(size):
        style_std = style_std.unsqueeze(dim=-1)

    content_mean, content_std = content_feat.mean(dim=[dim for dim in range(2, len(content_feat.shape))]), content_feat.std(dim=[dim for dim in range(2, len(content_feat.shape))])

    while len(content_mean.size()) < len(size):
        content_mean = content_mean.unsqueeze(dim=-1)
    while len(content_std.size()) < len(size):
        content_std = content_std.unsqueeze(dim=-1)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)