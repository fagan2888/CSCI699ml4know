import torch
import torch.nn.functional as F


def two_step_loss(score, label):
    """ Separate the "other" label. Use a binary classifier to detect "other" label and another
    18-way classifier to detect true labels.

    Args:
        true_label_score: tensor of shape (batch_size, 18)
        other_label_score: tensor of shape (batch_size, 2)
        label: (batch_size)

    Returns: loss

    """
    true_label_score, other_label_score = score
    not_other = label != 18

    if torch.cuda.is_available():
        binary_loss = F.cross_entropy(other_label_score, not_other.type(torch.cuda.LongTensor))
    else:
        binary_loss = F.cross_entropy(other_label_score, not_other.type(torch.LongTensor))

    # for class loss, we only consider those which is not "Other"
    true_label_score_not_other = true_label_score[not_other]
    not_other_label = label[not_other]
    class_loss = F.cross_entropy(true_label_score_not_other, not_other_label)
    return binary_loss + class_loss


def rank_loss(score, label, margin=1., gamma=1.):
    """ Santos(2015). Classifying Relations by Ranking with Convolutional Neural Networks. ACL.


    Args:
        score: (batch_size, 18)
        label: (batch_size)

    Returns: rank loss

    """
    # print(score.shape, label.shape)

    mask = label != 18  # create a mask and set the positive score for "Other" label to be zero.

    label_without_other = label.clone()
    label_without_other[mask == 0] = 0  # set other label to anything else

    score_label = torch.gather(score, dim=1, index=label_without_other.unsqueeze(-1)).squeeze(1)

    # print(score_label.shape)
    # print(mask.shape)

    # select negative label with maximum score
    _, top_2_index = torch.topk(score, k=2, dim=-1)
    # select the top 2 index that doesn't equal to label
    largest_index = top_2_index[:, 0]
    second_largest_index = top_2_index[:, 1]
    final_index = largest_index.clone()
    final_index[largest_index == label] = second_largest_index[largest_index == label]

    # print(final_index.shape)
    # print(final_index.type())

    negative_label_score = score.gather(dim=1, index=final_index.unsqueeze(-1)).squeeze(1)

    if torch.cuda.is_available():
        positive_score = torch.log1p(torch.exp(gamma * (margin - score_label))) * mask.type(torch.cuda.FloatTensor)
    else:
        positive_score = torch.log1p(torch.exp(gamma * (margin - score_label))) * mask.type(torch.FloatTensor)
    negative_score = torch.log1p(torch.exp(gamma * (-margin + negative_label_score)))

    # print(positive_score.shape, negative_score.shape)

    return (positive_score + negative_score).mean()


def rank_loss_classifier(score):
    """ Santos(2015). Classifying Relations by Ranking with Convolutional Neural Networks. ACL.
        if all the score are less than zero. Then, set it to 18. Otherwise, pick the largest one.

        Args:
            score: (batch_size, 18)

        Returns: classified label (batch_size)

    """
    label = torch.max(score, 1)[1].data
    mask = (score < 0).all(dim=1)
    label[mask == 1] = 18
    return label


def get_loss(type='cross_entropy'):
    if type == 'cross_entropy':
        return F.cross_entropy

    elif type == 'two_step':
        return two_step_loss
    elif type == 'rank':
        return rank_loss

    else:
        raise NotImplementedError


def cross_entropy_classifier(score):
    return torch.max(score, 1)[1].data


def two_step_classifier(score):
    true_label_score, other_label_score = score
    label = torch.max(true_label_score, 1)[1].data
    other_label = torch.max(other_label_score, 1)[1].data
    label[other_label == 0] = 18
    return label


def get_classifier(type='cross_entropy'):
    if type == 'cross_entropy':
        return cross_entropy_classifier
    elif type == 'two_step':
        return two_step_classifier
    elif type == 'rank':
        return rank_loss_classifier


def get_loss_classifier(type='cross_entropy'):
    return get_loss(type), get_classifier(type)
