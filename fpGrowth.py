# -*- coding:utf-8 -*-


def load_simpdat():
    simple_dat = [['r', 'z', 'h', 'j', 'p'],
                  ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                  ['z'],
                  ['r', 'x', 'n', 'o', 's'],
                  ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                  ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simple_dat


# 计数字典
def create_initset(dataset):
    ret_dict = {}
    for item in dataset:
        ret_dict[frozenset(item)] = ret_dict.get(frozenset(item), 0)+1
    return ret_dict


class TreeNode:
    def __init__(self, name_value, num_occur, parent_node):
        self.name = name_value
        self.count = num_occur
        self.node_link = None
        self.parent = parent_node
        self.children = {}

    def inc(self, num_occur):
        self.count += num_occur

    def disp(self, ind=1):
        print ' '*ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)


def create_tree(dataset, min_sup=1):
    header_table = {}
    for item in dataset:
        for w in item:
            header_table[w] = header_table.get(w, 0)+dataset[item]
    for k in header_table.keys():
        if header_table[k] < min_sup:
            del header_table[k]
    freq_itemset = set(header_table.keys())  # 过滤后的频繁集
    if len(freq_itemset) == 0:
        return None, None

    for k in header_table:
        header_table[k] = [header_table[k], None]
    ret_tree = TreeNode('Null Set', 1, None)

    for item, count in dataset.items():
        local_d = {}
        for w in item:
            if w in freq_itemset:
                local_d[w] = header_table[w][0]
        if len(local_d) > 0:
            ordered_items = [v[0] for v in sorted(local_d.items(), key=lambda d: d[1], reverse=True)]
            update_tree(ordered_items, ret_tree, header_table, count)
    return ret_tree, header_table


def update_tree(items, intree, header_table, count):
    if items[0] in intree.children:
        intree.children[items[0]].inc(count)
    else:
        intree.children[items[0]] = TreeNode(items[0], count, intree)  # 生成子节点
        if header_table[items[0]][1] is None:
            header_table[items[0]][1] = intree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], intree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1:], intree.children[items[0]], header_table, count)


def update_header(node_to_test, target_node):
    while node_to_test.node_link is not None:
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node











