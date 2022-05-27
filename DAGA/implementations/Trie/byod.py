#! /usr/bin/env python3
# -*- encoding: utf8 -*-

from argparse import ArgumentParser, FileType
from binascii import hexlify
from collections import Counter, OrderedDict, defaultdict, namedtuple
from csv import DictReader
from itertools import chain, count, repeat, product, tee, islice, cycle, accumulate
from math import log2, ceil, floor

ENCODING_MAP_INT_SIZE = 2


def windowed_iter(src, size):
    tees = tee(src, size)
    try:
        for i, t in enumerate(tees):
            for _ in range(i):
                next(t)
    except StopIteration:
        return zip([])
    return zip(*tees)


class MyCounter:
    def __init__(self, init_value=0):
        self.init_value = init_value
        self. value = init_value

    def __next__(self):
        self.value += 1

    def increase(self, n):
        self.value += n

    def reset(self):
        self.value = self.init_value


class NodeList:
    def __init__(self, degree, height, *, _depth=0):
        self.degree = degree
        self.height = height
        self.depth = _depth
        self.children = [None for _ in range(degree)]

    def init(self, model_it):
        for row in model_it:
            self.insert(row)

    def insert(self, row):
        if len(row) != (self.height - self.depth):
            raise ValueError(f'Invalid row length {len(row)} for node of depth {self.depth} within tree of height {self.height}')
        if len(row) == 0:
            return
        symbol = row[0]
        if not self.children[symbol]:
            self.children[symbol] = NodeList(degree=self.degree, height=self.height, _depth=self.depth + 1)
        self.children[symbol].insert(row[1:])

    def to_dict(self):
        d = OrderedDict()
        for i, v in enumerate(self.children):
            if v:
                d[i] = v.to_dict()
        return d

    def to_list(self):
        r = []
        for v in self.children:
            if v:
                l = v.to_list()
            else:
                l = None
            r.append(l)
        return r

    def inverted_list(self, depth, *, use_delta=False):
        result = {i: [] for i in range(self.degree)}
        self._inverted_list(result=result, depth=depth, parent_position=0, use_delta=use_delta)
        return result

    def _inverted_list(self, *, result, depth, parent_position, use_delta):
        if depth == self.depth:
            self._my_inverted_list(result=result, parent_position=parent_position, use_delta=use_delta)
            return
        for i, v in enumerate(self.children):
            if v:
                position = (parent_position * self.degree) + i
                v._inverted_list(result=result, depth=depth, parent_position=position, use_delta=use_delta)

    def _my_inverted_list(self, *, result, parent_position, use_delta):
        for i, v in enumerate(self.children):
            if v:
                position = parent_position
                if use_delta and len(result[i]) > 0:
                    position = position - result[i][-1] - 1
                result[i].append(position)

    def to_bitmaps(self, depth, *, include_empty=True, compressed=True):
        r = []
        empty_counter = MyCounter()
        self._to_bitmaps(r, depth=depth, include_empty=include_empty, compressed=compressed, empty_counter=empty_counter)
        return r

    def _to_bitmaps(self, r, *, depth=0, include_empty, compressed, empty_counter):
        if depth > 0:
            for v in self.children:
                if v:
                    v._to_bitmaps(r, depth=depth - 1, include_empty=include_empty, compressed=compressed, empty_counter=empty_counter)
                elif include_empty:
                    empty_amount = self.degree ** (depth - 1)
                    if compressed:
                        empty_counter.increase(empty_amount)
                    else:
                        r.extend([0] * empty_amount)
                    # self._empty_bitmap(r, height=height - 1)
            return
        b = 0
        for i, v in enumerate(self.children):
            if v:
                b |= 1 << i
        if compressed:
            if b == 0:
                empty_counter.increase(1)
            else:
                if empty_counter.value > 0:
                    r.append((0, empty_counter.value))
                    empty_counter.reset()
                r.append((1, b))
        else:
            r.append(b)
        return

    def pack(self, *, start_depth=1, end_depth=None, first_with_empty=False, delta_compression=False):
        if end_depth is None:
            end_depth = self.height
        if start_depth < 1 or end_depth > self.height or start_depth >= end_depth:
            raise ValueError(f'Invalid offsets {start_depth} {end_depth}')
        sizes = []
        indexes = []
        values = []
        last_floor = end_depth - 1
        first_floor = start_depth - 1
        # if height_start > 0:
        #     sizes.append(self.count_children(height=height_start-1))
        if not first_with_empty:
            sizes.append(self.count_children(height=first_floor))
        else:
            sizes.append(self.degree ** (first_floor + 1))
        for height in range(first_floor, last_floor):
            indexes_h = []
            values_h = []
            indexes.append(indexes_h)
            values.append(values_h)
            n = 0
            for v in self.children:
                n += v._pack(indexes_h, values_h, depth=height,
                             store_last_indexes = (height == last_floor),
                             include_empty = (first_with_empty and (height == first_floor)),
                             delta_compression=delta_compression)
            if n == 0:
                break
            sizes.append(n)
        return indexes, values

    def _pack(self, indexes, values, *, depth=0, store_last_indexes=False, include_empty=False, delta_compression=False):
        n = 0
        if depth > 0:
            for v in self.children:
                if v:
                    n += v._pack(indexes, values, depth=depth - 1,
                                 store_last_indexes=store_last_indexes, include_empty=include_empty,
                                 delta_compression=delta_compression)
                elif include_empty:
                    indexes.extend(repeat(0, self.degree ** (depth - 1)))
            return n
        ## if delta == 0
        if delta_compression:
            n = inverted_list_deltas(values, self.children)
        else:
            n = inverted_list(values, self.children)
        if not store_last_indexes:
            indexes.append(n)
        return n

    def pack_collapsed(self, *, height_start, first_with_empty=False):
        first_floor = height_start - 1
        s = set()
        d = self._collapse(height=first_floor, s=s)
        symbols_map = {v: i for i, v in enumerate(sorted(s))}
        indexes = []
        values = []
        self._count_collapsed(d, height=first_floor, symbols_map=symbols_map,
                              indexes=indexes, values=values)
        return symbols_map, indexes, values

    def _count_collapsed(self, d, *, height, symbols_map, indexes, values):
        if height==0:
            for k, v in sorted(d.items()):
                assert isinstance(v, tuple)
                values.extend(symbols_map[x] for x in v)
                indexes.append(len(v))
        else:
            for k, v in sorted(d.items()):
                self._count_collapsed(v, symbols_map=symbols_map, height=height-1, indexes=indexes, values=values)

    def _collapse(self, *, height, s):
        def ifinttuple(x):
            return x if not isinstance(x, int) else (x, )
        if height >= 0:
            d = {}
            for i, v in enumerate(self.children):
                if v:
                    r = v._collapse(height=height - 1, s=s)
                    if isinstance(r, frozenset):
                        s.update(r)
                        r = tuple(sorted(r))
                    d[i] = r
            return d
        # height < 0
        sc = frozenset(chain.from_iterable(v._collapse(height=height-1, s=s)
                                     for v in filter(None, self.children)))
        # for i, v in enumerate(self.children):
        #     if v:
        #         sc.update(v._collapse(height=height - 1))
        # if height > 0:
        #     return sc
        s = frozenset(i for i, v in enumerate(self.children) if v)
        if not sc:
            return s
        return frozenset((x, *ifinttuple(yz)) for x, yz in product(s, sc))

    def count_children(self, *, height=0):
        it = filter(None, self.children)
        if height == 0:
            return sum(1 for _ in it)
        return sum(v.count_children(height=height-1) for v in it)

    def stats(self, *, height=0):
        if height == 0:
            return self._stats()[1:]
        height -= 1
        children = self.collect_children(height)
        if not children:
            return None
        c = list(chain.from_iterable(v._stats()[0] for v in children if v))
        if not c:
            return None
        avg = sum(c) / len(c)
        ds = (sum((v - avg) ** 2 for v in c) / len(c)) ** (1 / 2)
        return avg, ds

    def collect_children(self, height):
        c = []
        self._collect_children(c, height=height)
        return c

    def _collect_children(self, c, *, height=0):
        if height == 0:
            c.extend(self.children)
            return
        for v in self.children:
            if v:
                v._collect_children(c, height=height - 1)

    def _stats(self):
        c = [sum(1 for b in v.children if b) for v in self.children if v]
        if not c:
            return [], 0, 0
        avg = sum(c) / len(c)
        ds = (sum((v - avg) ** 2 for v in c) / len(c)) ** (1 / 2)
        return c, avg, ds

    def check_sequence(self, sequence):
        if not sequence:
            return True
        if not self.children[sequence[0]]:
            return False
        return self.children[sequence[0]].check_sequence(sequence[1:])


VbData = namedtuple('VbData', 'encoding, max_value')


def vb01_encode(deltas, sizes, block_size=8, max_symbol=(2 ** 6) - 1):
    rle_values = {
        0: VbData(encoding=0b10000000, max_value=2 ** block_size),
        1: VbData(encoding=0b01000000, max_value=2 ** (block_size - 1))
    }
    c = 0
    r = []
    deltas_it = iter(deltas)
    try:
        dp = next(deltas_it)
    except StopIteration as e:
        raise ValueError('Empty delta values') from e
    for d in deltas_it:
        if d != dp and c > 0:
            yield rle_values[dp].encoding | (c - 1)
        if d not in rle_values:
            if d > max_symbol:
                raise ValueError(f'Invalid value "{d}": greater than max value {max_symbol}')
            yield d
            continue
        # d in rle_counters
        c += 1
        if c == rle_values[d].max_value:
            yield rle_values[d].encoding | (c - 1)
            c = 0
    return


def delta_size(deltas, sizes):
    for i, d, s in deltas:
        pass


def vb01_tree_encode(indexes, delta_values, indexes_sizes, nodes_sizes):
    deltas_size = 8
    indexes_skip = indexes
    values_skip = delta_values
    for depth, (i_size, v_size) in enumerate(zip(indexes_sizes, nodes_sizes)):
        offsets = [0]
        ip = 0
        for i in indexes_skip[:i_size]:
            offsets.append(i + ip)
            ip = i
        values_list = (values_skip[ip:i] for ip, i in zip(offsets, offsets[1:]))
        deltas_rle_all = []
        delta_sum = 0
        for nodes in values_list:
            if len(nodes[1:]) > 0:
                deltas_rle = run_length_with_skips(nodes[1:])
                deltas_rle_all.extend(deltas_rle)


def vb01_rle_encode(rle_seq):
    block_size = 8
    rle_values = {
        0: VbData(encoding=0b10000000, max_value=2 ** block_size),
        1: VbData(encoding=0b01000000, max_value=2 ** (block_size - 1))
    }
    r = []
    def encode_and_insert(n, v):
        if v not in rle_values:
            r.append([v] * (n + 1))
        else:
            e = n | rle_values[v]
            r.append(e)
    rle_it = iter(rle_seq)
    # the first value is expressed as a single symbol, regardless it being in rle_values
    n, v = next(rle_it)
    # now, use the normal encoding for the remainder of the run
    r.append(v)
    if n > 0:
        encode_and_insert(n - 1, v)
    for n, v in rle_it:
        encode_and_insert(n, v)
    return r


def inverted_list(result, input):
    n = 0
    for i, v in enumerate(input):
        if v:
            n += 1
            result.append(i)
    return n


def inverted_list_deltas(result, input):
    n = 0
    ip = -1
    for i, v in enumerate(input):
        if v:
            n += 1
            if ip == -1:
                result.append(i)
                ip = i
            else:
                result.append(i - ip - 1)
                ip = i
    return n


def check_trace(trace, n, tree):
    for i in range(len(trace) - n):
        sequence = trace[i:i + n]
        if not tree.check_sequence(sequence):
            return False
    return True


def run_length_with_skips(values, *, skip_number=None, skip_root_factor=None, counter_size=0, check_constraints=True):
    if skip_number is not None and skip_root_factor is not None:
        raise ValueError(f'Only one value can be assigned ({skip_number=}, {skip_root_factor=})')
    if skip_number is not None and skip_number < 0:
        raise ValueError
    if skip_root_factor is not None and (skip_root_factor <= 0 or skip_root_factor >=1):
        raise ValueError
    r_skips = []
    values_skip = []
    if skip_number is not None:
        skip_size = floor(len(values) / skip_number)
        if check_constraints and skip_size <= skip_number:
            raise ValueError('Invalid skip number: lower than the square of the number of values')
        check_skip = lambda skip_progress: skip_progress < skip_size
    elif skip_root_factor is not None:
        skip_size = floor(len(values) ** skip_root_factor)
        check_skip = lambda skip_progress: skip_progress < skip_size
    else:
        skip_number = 0
        skip_size = 0
        check_skip = lambda skip_progress: True
    if counter_size > 0:
        max_run = (2 ** counter_size) - 1
        check_progress = lambda v, vp, n: n < max_run and v == vp
    else:
        check_progress = lambda v, vp, n: v == vp
    r = []
    it = iter(values)
    vp = next(it)
    run_progress = 0
    skip_progress = 0
    run_number = 0
    count_values = 0
    for v in it:
        skip_progress += 1
        count_values += vp
        if check_progress(v, vp, run_progress) and check_skip(skip_progress):
            run_progress += 1
        else:
            r.append((run_progress, vp))
            vp = v
            run_number += 1
            run_progress = 0
            if not check_skip(skip_progress):
                r_skips.append(run_number)
                values_skip.append(count_values)
                count_values = 0
                run_number = 0
                skip_progress = 0
    if skip_root_factor is None:
        assert len(r_skips) == skip_number, f'Unexpected number of index skips {len(r_skips)} != {skip_number} (counter_size={counter_size})'
        assert len(values_skip) == skip_number, f'Unexpected number of values skips {len(values_skip)} != {skip_number} (counter_size={counter_size})'
    r.append((run_progress, vp))
    return r, (skip_size, r_skips, values_skip)


def mask(i):
    return (1 << i) - 1


def notmask(block_size, i):
    return mask(block_size) - mask(i)


def extract(v1, v2, offset, symbol_size, block_bitsize):
    rightpad = block_bitsize - symbol_size - offset
    v = v1 & mask(block_bitsize - offset)
    if rightpad == 0:
        return v
    if rightpad > 0:
        v >>= rightpad
        return v
    # note that rightpad is negative in the next lines, thus (-rightpad) is positive
    v <<= (-rightpad)
    v |= (v2 >> (block_bitsize + rightpad))
    return v




def unpack_trace_multi(trace, symbols_sizes, block_bitsize, *, size=None):
    r = []
    offset = 0
    trace_it = iter(trace)
    tp = next(trace_it)
    t = next(trace_it, None)
    if size:
        it = range(size)
    else:
        it = count()
    for _, s in zip(it, cycle(symbols_sizes)):
        v = extract(tp, t, offset, s, block_bitsize)
        r.append(v)
        offset += s
        if t is None and (offset + s) >= block_bitsize:
            break
        if offset < block_bitsize:
            continue
        offset -= block_bitsize
        if t is None:
            break
        tp = t
        t = next(trace_it, None)
        if t is None and (offset + s) > block_bitsize:
            break
    return r


def unpack_trace(trace, symbol_size, block_bitsize, *, size=None):
    return unpack_trace_multi(trace, symbols_sizes=[symbol_size], block_bitsize=block_bitsize, size=size)


def build_model(*data_lists, size):
    freq = Counter(chain.from_iterable(data_lists))
    it = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    decode_map = {k: i for i, (k, v) in enumerate(it)}
    model = sorted(
                chain.from_iterable(
                    frozenset(tuple(decode_map[v] for v in data[i:i + size]) for i in range(len(data) - size))
                    for data in data_lists
                )
            )
    return decode_map, model


def pack_trace_multi_it(trace_it, symbols_sizes, *, block_bytesize=1):
    n_symbols = [2 ** s for s in symbols_sizes]
    # symbol_size = ceil(log2(n_symbols))
    # print("symbol size is", symbol_size)
    block_bitsize = block_bytesize * 8
    for s in symbols_sizes:
        if s > block_bitsize:
            raise ValueError(f'Symbol size {s} exceeds pack size {block_bitsize}')
    # r = []
    offset = 0
    vp = 0
    for v, s, ns in zip(trace_it, cycle(symbols_sizes), cycle(n_symbols)):
        if v >= ns:
            raise ValueError(f'Invalid symbol {v}: exceeds number of symbols for symbol size {s}')
        missing_bits = block_bitsize - offset
        store_bit = s - missing_bits
        offset = (offset + s) % block_bitsize
        if store_bit < 0:
            vp |= (v << -store_bit)
            continue
        vp |= (v >> store_bit)
        yield vp
        # r.append(vp)
        if offset == 0:
            vp = 0
        else:
            vp = ((v & ~(0b1 << store_bit)) << (block_bitsize - s + missing_bits)) & ((0b1 << block_bitsize) - 1)
    if offset != 0:
        yield vp


def pack_trace_it(trace_it, symbol_size, *, block_bitsize=8):
    return pack_trace_multi_it(trace_it, [symbol_size], block_bytesize=block_bitsize // 8)


def pack_trace(trace_it, n_symbols, *, block_bitsize=8):
    return list(pack_trace_it(trace_it, n_symbols, block_bitsize=block_bitsize))


def compress_bitmaps_offset(bitmaps, size, height, offset):
    offsets = [offset]
    for _ in range(height):
        offsets.append(offsets[-1]//size)
    #print(offsets)
    for i in range(height):
        bitmaps = bitmaps[size**i:]
        for b, o in zip(bitmaps, range(offsets[height - i])):
            if not b:
                #print(i, o)
                offset -= size ** (height - i - 1)
    return offset


def count_compressed(it, index):
    counter = 0
    i = 0
    for c, v in it:
        counter += c*v
        i += c
        if i >= index:
            break
    else:
        raise ValueError(f'Index {index} exceeds sequence length {counter}')
    return counter, i - index


def sparse_uncompressed_lookup(a, seek):
    c = sum(a[:seek])
    return c, a[seek], seek


def sparse_compressed_lookup(values, seek):
    c = 0
    a0_it = iter(values[::2])
    a1_it = iter(values[1::2])
    diff = 0
    tot = 0
    i = 0
    for i, (n, v) in enumerate(zip(a0_it, a1_it), 1):
        tot += (n + 1)
        diff = tot - seek
        if diff >= 0:
            c += (n + 1 - diff) * v
            break
        c += (n + 1) * v
    if diff > 0:
        # if diff > 0 v is always defined
        v_next = v
    else:
        v_next = next(a1_it)
    return c, v_next, (i + 1) * 2


def to_invertedlists(values):
    d = defaultdict(list)
    for i, v in enumerate(values):
        d[v].append(i)
    return d


def stats(positions):
    deltas = []
    for vp, v in zip(positions, positions[1:]):
        deltas.append(v - vp)
    deltas.sort()
    n = len(deltas)
    median = deltas[n // 2]
    mean = sum(deltas) / n
    return median, mean


def check_sequence_counter_0(sequence, counters, values, *, compressed_sizes=False,
                             first_skip, skip_number, skips):
    first_value = sequence[0]
    return check_sequence_counter(sequence[1:], indexes_seek=first_value, counters=counters, values=values,
                                  compressed_sizes=compressed_sizes, _sequence_offset=1,
                                  first_skip=first_skip, skip_number=skip_number, skips=skips)


def check_sequence_counter(sequence, indexes_seek, counters, values, *,
                           compressed_sizes=False, raise_error=True, verbose=False, _sequence_offset=0,
                           first_skip, skip_number, skips):
    if len(counters) != len(values):
        raise ValueError('Index and values sizes do not have the same length')

    if not compressed_sizes:
        indexes_lookup = sparse_uncompressed_lookup
    else:
        indexes_lookup = sparse_compressed_lookup

    i = 0
    for i, (symbol, counters_i, values_i) in enumerate(zip(sequence, counters, values), 1):
        # if indexes_seek > floor_size_indexes:
        #     raise ValueError(f'Invalid sequence: index seek {indexes_seek} exceeds allowed size at depth {i}')
        # if verbose:
        #     print(f'\n\nfloor {i} symbol is {symbol} sizes are {floor_size_indexes} {floor_size_values}, index_seek is {indexes_seek}')
        # print(f'first indexes are {indexes[:12]}')
        # print(f'first values are {values[:10]}')
        # #print(indexes[:intrabranch_seek])
        # print(f'counting on indexes to index {seek_branch + seek_node}: {sum(indexes[:seek_branch + seek_node])}')
        # print(f'inter is {seek_branch}, intra is {seek_node}')
        # indexes_seek = seek_branch + seek_node
        # use_skip = True
        if skip_number != 0 and i > first_skip:
        # if skip_number > 0 and skip_size > 0:
            skips_jump, skips_indexes, skips_data = skips[i - first_skip - 1]
            # assert len(skips_indexes) == skip_number, f"Invalid length of index skip positions {len(skips_indexes)} for depth {i}"
            # assert len(skips_data) == skip_number, f"Invalid length of values skip positions {len(skips_data)} for depth {i}"
            skips_offset = indexes_seek // skips_jump
            skip_value = skips_jump * skips_offset
            skip_index = sum(skips_indexes[:skips_offset]) if skips_offset > 0 else 0
            skip_branch = sum(skips_data[:skips_offset]) if skips_offset > 0 else 0
        else:
            skip_value = 0
            skip_index = 0
            skip_branch = 0
        seek_branch, branch_size, consumed_indexes = indexes_lookup(counters_i[skip_index * 2:], indexes_seek - skip_value)
        assert branch_size > 0, "Invalid branch size value zero"
        if skip_number != 0 and i > first_skip:
            seek_branch += skip_branch
            consumed_indexes += skip_index
        # indexes = indexes[indexes_seek:]
        # branch_size = indexes[0]
        # indexes = indexes[consumed_indexes:]
        if verbose:
            print(f'-- consumed indexes is {consumed_indexes}')
            print(f'-- next branch size is {branch_size}')
            print(f'-- seeking {seek_branch} on values')
        # values = values_i[seek_branch:]
        # print(f'searching symbol {symbol} in {values[:branch_size]}')
        node = values_i[seek_branch:seek_branch + branch_size]
        try:
            seek_node = node.index(symbol)
        except ValueError as e:
            raise ValueError(f'Invalid sequence: cannot find symbol {symbol} at depth {i + _sequence_offset} in values {node}', symbol, i + _sequence_offset) from e
        indexes_seek = seek_node + seek_branch
        if verbose:
            print(f'-- new index seek value is {indexes_seek}')
        if i == len(counters) - 1:
            i += 1
            break
        # print(f'-- seeking {floor_size_values - seek_branch} on values')
        # values = values[floor_size_values - seek_branch:]
        # print(f'seeking {floor_size_indexes - indexes_seek - 1} on indexes')
        # indexes = indexes[floor_size_indexes:]
        # indexes = indexes[floor_size_indexes - consumed_indexes:]
        # print(f'{i:2d}: Intra seek: {seek_node:3d}, branch size: {branch_size:4d}')
    else:
        if raise_error:
            raise ValueError(f'Sequence finished before expected', i)
    return i + _sequence_offset, indexes_seek


def check_bitmaps_0(sequence, bitmaps, n_symbols, sequence_size):
    if sequence[0] >= n_symbols:
        raise ValueError
    return check_bitmaps(sequence[1:], bitmaps, size=n_symbols, sequence_size=sequence_size, offset=sequence[0], height=1)


def check_bitmaps(sequence, bitmaps, size, sequence_size, *, offset, height):
    def check(b, v):
        return b & (1 << v) != 0
    i = 0
    for i, v in zip(range(height, sequence_size + height), sequence):
        # print(i)
        if len(bitmaps) < size**i:
            raise ValueError('Invalid bitmap length', i, len(bitmaps))
        if not check(bitmaps[offset], v):
            raise ValueError('Bitmap verification failed', v, i, offset, len(bitmaps), bitmaps[offset])
        offset *= size
        offset +=  v
        bitmaps = bitmaps[size ** i:]
    return i, offset

#### CLI-RELATED CODE

def int_min(n):
    def func(s):
        try:
            v = int(s)
        except ValueError:
            raise ValueError(f'Invalid value \'{s}\': expected an integer greater or equal than {n}')
        if v<n:
            raise ValueError(f'Invalid value \'{s}\': expected an integer greater or equal than {n}')
        return v
    return func


def compute_encoding_map(traces_it):
    print('Computing symbols frequencies to compute encoding map')
    freq = Counter(traces_it)
    sorted_symbols = [k for k, v in sorted(freq.items(), key=lambda x: (x[1], x[0]), reverse=True)]
    encoding_map = OrderedDict({s: i for i, s in enumerate(sorted_symbols)})
    return encoding_map


def write_encoding_map(encoding_out, encoding_map):
    for s, i in encoding_map.items():
        encoding_out.write(s.to_bytes(ENCODING_MAP_INT_SIZE, 'little'))


def read_encoding_map(encoding_in):
    print('Using provided encoding map')
    encoding_map = {}
    for i in count():
        v = encoding_in.read(ENCODING_MAP_INT_SIZE)
        if len(v) < ENCODING_MAP_INT_SIZE:
            break
        encoding_map[int.from_bytes(v, 'little')] = i
    return encoding_map


def get_traces_it(traces_files, *, show_progress=False):
    def rewind():
        for f in traces_files:
            f.seek(0)
    tqdm = lambda x, total: x
    size = None
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            print("Cannot show progress bar: tqdm missing")
        else:
            size = files_size(traces_files, lines=True)
    extract_it = ((int(r['CAN_ID'], 16) for r in DictReader(f, delimiter=',')) for f in traces_files)
    it = chain.from_iterable(extract_it)
    if tqdm is not None:
        _it = it
        it = tqdm(_it, total=size)
    return it, rewind


def read_or_compute_encoding_map(traces_files, *, encoding_in, encoding_out):
    if encoding_in is None:
        it, rewind = get_traces_it(traces_files, show_progress=True)
        encoding_map = compute_encoding_map(it)
        rewind()
        if encoding_out:
            write_encoding_map(encoding_out=encoding_out, encoding_map=encoding_map)
    else:
        encoding_map = read_encoding_map(encoding_in)
    return encoding_map


def write_array(values, fout, block_size):
    for v in values:
        fout.write(v.to_bytes(block_size, 'little'))


def write_unaligned(values, fout, *, symbol_size, block_size):
    it = pack_trace_it(values, symbol_size, block_bitsize=block_size * 8)
    size = 0
    for v in it:
        size += block_size
        fout.write(v.to_bytes(block_size, 'little'))
    return size


def write_encoded_multi(values, fout, *, symbols_sizes, block_size, bs_size):
    data = list(pack_trace_multi_it(values, symbols_sizes, block_bytesize=block_size))
    data_size = len(data)
    size = 0
    for v in data:
        size += block_size
        fout.write(v.to_bytes(block_size, 'little'))
    return size

def get_tqdm():
    try:
        from tqdm import tqdm
    except:
        print("Install tqdm to get progress bars")
        tqdm = lambda x, *args, **kwargs: x
    return tqdm


def read_sparse_model(*, size, counters_in, values_in, block_size, bs_size):
    c_sizes, counters = read_matrix_plain(counters_in, matrix_size=size, row_size_bytes=bs_size, values_size=block_size)
    v_sizes, values = read_matrix_plain(values_in, matrix_size=size, row_size_bytes=bs_size, values_size=block_size)
    return counters, values


def read_bitmaps(fin, *, block_size, n_symbols, n):
    it = file_it(fin, block_size)
    bitmaps = unpack_trace(it, symbol_size=n_symbols, block_bitsize=block_size*8)
    return bitmaps


def verify_bitmap(sequences_files, *, format_in, bitmaps_in, window_size, block_size_trace, block_size_bitmaps):
    n_symbols = int.from_bytes(format_in.read(1), 'little')
    _window_size = int.from_bytes(format_in.read(1), 'little')
    if _window_size != window_size:
        raise ValueError(f'Mismatch between window size values: CLI input -> {window_size}, schema file -> {_window_size}')
    # bitmaps = list(unpack_trace(file_it(bitmaps_in, block_size_bitmaps), symbol_size=n_symbols, block_size=block_size_bitmaps*8))
    bitmaps = read_bitmaps(bitmaps_in, block_size=block_size_bitmaps, n_symbols=n_symbols, n=window_size-1)
    symbol_size = ceil(log2(n_symbols))
    def decode_f(f):
        return unpack_trace(file_it(f, block_size_trace), symbol_size=symbol_size, block_bitsize=block_size_trace*8)
    decoded_trace = [list(decode_f(f)) for f in sequences_files]
    trace_it = chain.from_iterable(windowed_iter(d, window_size) for d in decoded_trace)
    size = files_size(sequences_files)
    for i, s in enumerate(get_tqdm()(trace_it, total=size)):
        check_bitmaps_0(sequence=s, bitmaps=bitmaps, n_symbols=n_symbols, sequence_size=window_size)


def verify_sparse_model(sequences_files, *, format_in, skips_in, values_in, counters_in, block_size, rle_size):
    n_symbols = int.from_bytes(format_in.read(1), 'little')
    window_size = int.from_bytes(format_in.read(1), 'little')
    print(f'Schema: {n_symbols=} , {window_size=}')
    # if _window_size != window_size:
    #     raise ValueError(f'Mismatch between window size values: CLI input -> {window_size}, schema file -> {_window_size}')
    row_sizes, skips_number, skips_start, skips_jumps, skips_indexes, skips_data = read_skips(skips_in)
    skips = list(zip(skips_jumps, skips_indexes, skips_data))
    symbol_size = ceil(log2(n_symbols))
    if rle_size is None:
        rle_size = symbol_size
    def decode_f(f):
        return unpack_trace(file_it(f, block_size), symbol_size=symbol_size, block_bitsize=block_size*8)
    # def decode_l(l):
    #     return
    decoded_trace = [list(decode_f(f)) for f in sequences_files]
    trace_it = chain.from_iterable(windowed_iter(d, window_size) for d in decoded_trace)
    encoded_counters, encoded_values = read_sparse_model(size=window_size-1, counters_in=counters_in,
                                                        values_in=values_in, block_size=block_size, bs_size=4)
    counters = [list(unpack_trace_multi(i, symbols_sizes=[rle_size, symbol_size], block_bitsize=block_size*8)) for i in encoded_counters]
    values = [list(unpack_trace(v, symbol_size=symbol_size, block_bitsize=block_size*8)) for v in encoded_values]
    size = files_size(sequences_files)
    for i, s in enumerate(get_tqdm()(trace_it, total=size)):
        if s[0] >= n_symbols:
            raise ValueError()
        r, _ = check_sequence_counter_0(s, counters=counters, values=values,
                                        compressed_sizes=True, first_skip=skips_start,
                                        skip_number=skips_number, skips=skips)
        if r != window_size:
            raise ValueError(f'Sequence {i} mismatch at position {r}')


def byte_size(v):
    return int(log2(v) // 8) + 1


def mean(values):
    return sum(values)/len(values)


def create_sparse_model(sequences_files, window_size, *, values_out, counters_out,
                 block_size, format_out, n_symbols, skips_size, skips_root_factor, skips_out, rle_size):

    symbol_size, all_sequences, tree = load_model(sequences_files=sequences_files, window_size=window_size,
                                                  block_size=block_size, n_symbols=n_symbols)

    print('Creating model indexes')
    indexes, values = tree.pack(first_with_empty=False)
    indexes_compressed = []
    # indexes_sizes_compressed = []
    # indexes_compressed_packed = []
    idx = indexes
    # offsets to the values of the indexes (built as delta encoding)
    skips_indexes = []
    # offsets to the values of the data (built as delta encoding)
    skips_data = []
    # size of the indexes "skipped" by each skip value
    skips_jumps = []
    # first depth at which we use skips values
    skips_start = -1
    # number of indexes on which we use skip values
    skips_number = 0
    # skips_sizes = []
    if rle_size is None:
        rle_size = symbol_size
    for i, indexes_i in enumerate(indexes):
        _skips_size = None
        size = len(indexes_i)
        if skips_size is not None and skips_size > 0 and skips_size ** 2 <= size:
            _skips_size = skips_size
            if skips_number == 0:
                skips_start = i
            skips_number += 1
        elif skips_root_factor is not None:
            if skips_number == 0:
                skips_start = i
            skips_number += 1
        rle_data = run_length_with_skips(indexes_i, skip_number=_skips_size, skip_root_factor=skips_root_factor, counter_size=rle_size)
        idx_compressed_packed, (floor_skips_size, floor_skips_positions, floor_values_skips) = rle_data
        print(f'{i+1:2d}: Avg run: {mean([l for l, _ in idx_compressed_packed]):3.2f}')
        if _skips_size != 0:
            skips_jumps.append(floor_skips_size)
            skips_indexes.append(floor_skips_positions)
            skips_data.append(floor_values_skips)
        # indexes_compressed_packed.append(idx_compressed_packed)
        # idx_compressed = list(chain.from_iterable(idx_compressed_packed))
        indexes_compressed.append(idx_compressed_packed)
        # indexes_sizes_compressed.append(len(idx_compressed_packed))
        # idx = idx[size:]
    print('Writing model indexes')
    write_sparse_model(counters=indexes_compressed, values=values,
                block_size=block_size, bs_sizes=4, symbol_size=symbol_size, counters_out=counters_out,
                values_out=values_out, rle_size=rle_size)
    write_format(format_out, n_symbols=n_symbols, sequence_size=window_size)
    write_skips(skips_out, skips_start=skips_start, skips_number=skips_number,
                 skips_jumps=skips_jumps, skips_indexes=skips_indexes, skips_data=skips_data)


def write_sparse_model(*, counters, values, symbol_size, block_size, bs_sizes,
                counters_out, values_out, rle_size):
    encoded_counters = []
    for idx in counters:
        idx_flat = chain.from_iterable(idx)
        l = list(pack_trace_multi_it(idx_flat, symbols_sizes=[rle_size, symbol_size], block_bytesize=block_size))
        encoded_counters.append(l)
    write_matrix_plain(counters_out, matrix=encoded_counters, block_size=block_size, sizes_size=bs_sizes)
    encoded_values = []
    for v in values:
        l = list(pack_trace_it(v, symbol_size, block_bitsize=block_size * 8))
        encoded_values.append(l)
    write_matrix_plain(values_out, matrix=encoded_values, block_size=block_size, sizes_size=bs_sizes)


def write_format(fout, n_symbols, sequence_size):
    int_to_file(fout, n_symbols, 1)
    int_to_file(fout, sequence_size, 1)


def read_format(fin):
    n_symbols = int_from_file(fin, 1)
    sequence_size = int_from_file(fin, 1)
    return n_symbols, sequence_size


def write_skips_plain(fout, skips_number, skips_start, skips_jumps, skips_indexes, skips_data):
    int_to_file(fout, skips_number, 1)
    if skips_number == 0:
        return
    int_to_file(fout, skips_start, 1)
    for j in skips_jumps:
        int_to_file(fout, j, 2)
    row_sizes = write_matrix_plain(fout, skips_indexes, sizes_size=4, block_size=2)
    print(f'Skips sizes are {row_sizes}')
    write_matrix_plain(fout, skips_data, row_sizes=row_sizes, sizes_size=4, block_size=2)


def read_skips_plain(fin):
    skips_number = int_from_file(fin, 1)
    if skips_number == 0:
        return skips_number, None, None, None, None
    skips_start = int_from_file(fin, 1)
    skips_jumps = [ int_from_file(fin, 2) for _ in range(skips_number)]
    row_sizes, skips_indexes = read_matrix_plain(fin, matrix_size=skips_number, values_size=2, row_size_bytes=4)
    _, skips_data = read_matrix_plain(fin, matrix_size=skips_number, row_sizes=row_sizes, values_size=2, row_size_bytes=4)
    return row_sizes, skips_number, skips_start, skips_jumps, skips_indexes, skips_data

# def write_schema_struct(fout, skips_start, skip_number, skips_jumps, skips_indexes, skips_data):
#     fout.write(skip_number.to_bytes(1, 'little'))
#     if skip_number == 0:
#         return
#     fout.write(skips_start.to_bytes(1, 'little'))
#     fout.write(len(skips_jumps).to_bytes(1, 'little'))
#     for skip_size in skips_jumps:
#         fout.write(skip_size.to_bytes(2, 'little'))
#     write_matrix_struct(fout, skips_indexes)
#     write_matrix_struct(fout, skips_data)


write_skips = write_skips_plain
read_skips = read_skips_plain


# number of bytes to store the row size
# ROW_SIZE_BYTES=4


def write_matrix_plain(f, matrix, *, row_sizes=None, sizes_size, block_size):
    if row_sizes is None:
        # int_to_file(f, len(matrix), matrix_size_bytes)
        row_sizes = []
        size = len(matrix[0])
        row_sizes.append(size)
        int_to_file(f, size, sizes_size)
        # size_p = size
        for row in matrix[1:]:
            size = len(row)
            row_sizes.append(size)
            int_to_file(f, size, sizes_size)
            # int_to_file(f, size - size_p, sizes_size)
            # size_p = size
    for row in matrix:
        for s in row:
            f.write(s.to_bytes(block_size, 'little'))
    return row_sizes


def read_matrix_plain(f, matrix_size, *, row_sizes=None, row_size_bytes, values_size):
    matrix = []
    if row_sizes is None:
        # size = int_from_file(f, matrix_size_bytes)
        row_sizes = [int_from_file(f, row_size_bytes) for _ in range(matrix_size)]
        # row_sizes_diff = [int_from_file(f, row_size_bytes) for _ in range(matrix_size)]
        # row_sizes = list(accumulate(row_sizes_diff))
    for ncolumns in row_sizes:
        row = []
        matrix.append(row)
        for _ in range(ncolumns):
            data = f.read(values_size)
            row.append(int.from_bytes(data, 'little'))
    return row_sizes, matrix


def int_from_file(f, size):
    return int.from_bytes(f.read(size), 'little')


def int_to_file(f, v, size):
    f.write(v.to_bytes(size, 'little'))


def write_row_struct(f, row, *, block_size=1):
    size = len(row)
    base = min(row)
    delta_bits = min(v - base for v in row).bit_length()
    f.write(size.to_bytes(2, 'little'))
    f.write(delta_bits.to_bytes(2, 'little'))
    f.write(base.to_bytes(2, 'little'))
    data_it = pack_trace_it((v - base for v in row), delta_bits, block_bitsize=block_size * 8)
    for v in data_it:
        f.write(v.to_bytes(block_size, 'little'))


def read_row_struct(f, *, block_size=1):
    size = int.from_bytes(f.read(2), 'little')
    delta_bits = int.from_bytes(f.read(2), 'little')
    base = int.from_bytes(f.read(2), 'little')
    data_it = (int.from_bytes(f.read(block_size), 'little') for _ in range(ceil(delta_bits * size / (block_size * 8))))
    return [base + v for v in unpack_trace(data_it, symbol_size=delta_bits, block_bitsize=block_size*8)]


def write_matrix_struct(f, matrix, *, block_size=1):
    size = len(matrix)
    f.write(size.to_bytes(2, 'little'))
    for row in matrix:
        write_row_struct(f, row, block_size=block_size)


def read_matrix_struct(f):
    m = []


write_matrix = write_matrix_plain
read_matrix = read_matrix_plain
# write_matrix = write_matrix_encoded
# read_matrix = read_matrix_encoded
# write_matrix = write_matrix_struct
# read_matrix = read_matrix_struct

def file_it(f, n):
    while True:
        v = f.read(n)
        if len(v) < n:
            return
        yield int.from_bytes(v, 'little')


def files_size(files, *, lines=False):
    if lines:
        size = sum(1 for f in files for _ in f)
    else:
        size = sum(f.seek(0, 2) for f in files)
    for f in files:
        f.seek(0)
    return size


def encode_trace(traces_files, block_size, trace_out, *, encoding_out=None, encoding_in=None, load=False):
    if encoding_out is None and encoding_in is None:
        raise ValueError('missing both encoding out and encoding in')
    encoding_map = read_or_compute_encoding_map(traces_files,
                                                encoding_in=encoding_in, encoding_out=encoding_out)

    print('Encoding map is')
    for k, v in encoding_map.items():
        print(f'{k:3d}: {v:3d}')
    if not trace_out:
        print('No output specified for encoded trace')
        return

    print('Writing encoded trace')
    it, rewind = get_traces_it(traces_files, show_progress=True)
    encoded_trace_it = (encoding_map[t] for t in it)
    symbol_size = ceil(log2(len(encoding_map)))
    write_unaligned(encoded_trace_it, fout=trace_out, symbol_size=symbol_size, block_size=block_size)


def float_range(min_value, max_value, *, exclude=True):
    def func(s):
        try:
            v = float(s)
        except ValueError as e:
            raise ValueError(f'Invalid float value {s}') from e
        if exclude:
            if v < min_value or v > max_value:
                raise ValueError(f'Invalid value {s}: must be within range ]{min_value}:{max_value}[')
            elif v <= min_value or v >= max_value:
                raise ValueError(f'Invalid value {s}: must be within range [{min_value}:{max_value}]')
        return v
    return func


def matrix_to_c(m, bs):
    ms = [[v.to_bytes(bs, 'big') for v in row] for row in m]
    s = ',\n\t'.join(','.join(f'0x{hexlify(vs).decode()}U' for vs in rows) for rows in ms)
    return f'{{\n\t{s}\n}}'


def array_to_c(values, bs):
    a = [v.to_bytes(bs, 'big') for v in values]
    s = ','.join(f'0x{hexlify(v).decode()}U' for v in a)
    return f'{{\n\t{s}\n}}'


def array_size(a):
    v = len(a)
    if v == 0:
        raise ValueError('Invalid empty array')
    return v


def matrix_size(m):
    return sum(map(array_size, m))


c_template_include = '''
#ifdef __cplusplus
extern "C" {
#endif
#include "types.h"
#ifdef __cplusplus
};
#endif

// if PROGMEM is not defined, define a dummy empty one to use simpler code later 
#ifdef __USE_AVR_FLASH
    #include <avr/pgmspace.h>
    #define _PROGMEM PROGMEM
#else
    #define _PROGMEM
#endif

'''


c_template_skips = lambda *, jumps, sizes, indexes, data, number, start: f"""
uint16_t _skips_jumps[{array_size(jumps)}] = {array_to_c(jumps, 2)};
uint32_t _skips_sizes[{array_size(sizes)}] = {array_to_c(sizes, 4)};
uint16_t _skips_indexes[{matrix_size(indexes)}] = {matrix_to_c(indexes, 2)};
uint16_t _skips_data[{matrix_size(data)}] = {matrix_to_c(data, 2)}; 

Skips skips = {{
    .number = {number},
    .start = {start},
    .jumps = _skips_jumps,
    .sizes = _skips_sizes,
    .counters = _skips_indexes,
    .values = _skips_data
}};
"""


c_template_schema = lambda *, n_symbols, window_size: f"""
Schema schema = {{
    .n_symbols = {n_symbols},
    .window_size = {window_size}
}};
"""


def c_template_srlemodel(*, counters, values, block_size):
    # counters_sizes = [len(c) for c in counters]
    # values_sizes = [len(v) for v in values]
    get_counters_s = lambda i, a: f"const block_type _counters_{i}[{array_size(a)}] = {array_to_c(a, block_size)};"
    get_values_s = lambda i, a: f"const block_type _values_{i}[{array_size(a)}] _PROGMEM = {array_to_c(a, block_size)};"
    get_vars_names = lambda n, c: f"{','.join(f'_{n}_{i}' for i in range(len(c)))}"

    c_s = '\n'.join(get_counters_s(i, c) for i, c in enumerate(counters))
    v_s = '\n'.join(get_values_s(i, v) for i, v in enumerate(values))

    s = f"""
{c_s}
{v_s}

const counters_t* _counters[{array_size(counters)}] = {{ {get_vars_names('counters', counters)} }};
const values_t* _values[{array_size(values)}] = {{ {get_vars_names('values', values)} }};

SRLEModel model = {{
    .counters = _counters,
    .values = _values,
}};
"""
    return s


# '''
# const block_type _model_data[{array_size(values)}] _PROGMEM = {array_to_c(values, block_size)};
# const size_type _model_indexes_sizes[{array_size(indexes_sizes)}] = {array_to_c(indexes_sizes, 4)};
# const size_type _model_data_sizes[{array_size(data_sizes)}] = {array_to_c(data_sizes, 4)};

c_template_trace = lambda *, trace, block_size: f"""
#define TRACE_SIZE {array_size(trace)}
const mysize_t trace_size = TRACE_SIZE;
const block_type trace[TRACE_SIZE] = {array_to_c(trace, block_size)};

#undef TRACE_SIZE

"""


def model_to_c(*, format_in, skips_in, values_in, counters_in, block_size, data_out):
    row_sizes, skips_number, skips_start, skips_jumps, skips_indexes, skips_data = read_skips(skips_in)
    data_out.write(c_template_include)
    data_out.write(c_template_skips(jumps=skips_jumps, number=skips_number, start=skips_start, indexes=skips_indexes,
                               sizes=row_sizes, data=skips_data))
    n_symbols, window_size = read_format(format_in)
    data_out.write(c_template_schema(n_symbols=n_symbols, window_size=window_size))
    counters, values = read_sparse_model(counters_in=counters_in, size=window_size-1, values_in=values_in,
                                                         block_size=block_size, bs_size=4)
    data_out.write(c_template_srlemodel(counters=counters, values=values, block_size=block_size))


def trace_to_c(*, trace_in, trace_out, block_size, limit=None):
    if limit is None:
        it = file_it(trace_in, block_size)
    else:
        it = islice(file_it(trace_in, block_size), limit)
    trace = list(it)
    trace_out.write(c_template_include)
    trace_out.write(c_template_trace(block_size=block_size, trace=trace))


def load_model(sequences_files, window_size, n_symbols, block_size):
    try:
        from tqdm import tqdm
    except ImportError:
        print("Cannot show progress bar: tqdm missing")
        tqdm = lambda x, s: x
    symbol_size = ceil(log2(n_symbols))
    size = files_size(sequences_files)
    def decode(f):
        return unpack_trace(file_it(f, 1), symbol_size=symbol_size, block_bitsize=block_size * 8)
    it = chain.from_iterable(windowed_iter(decode(f), window_size) for f in sequences_files)
    print('Loading sequences and creating flat model')
    all_sequences = sorted(set(tqdm(it, total=size)))
    print('Creating model tree')
    tree = NodeList(degree=n_symbols, height=window_size)
    tree.init(tqdm(all_sequences))
    print('Getting tree statistics')
    for i in range(window_size - 1):
        avg, devstd = tree.stats(height=i)
        print(f'{i+1:2d}: avg={avg:2.2f} , devstd={devstd:2.2f}')
    return symbol_size, all_sequences, tree


def create_bitmap(sequences_files, window_size, n_symbols, block_size_trace, block_size_bitmaps, bitmaps_out, format_out):
    symbol_size, all_sequences, tree = load_model(sequences_files=sequences_files, window_size=window_size, n_symbols=n_symbols,
                                                  block_size=block_size_trace)
    bitmaps = []
    for d in range(1, window_size):
        bitmaps.extend(tree.to_bitmaps(depth=d, include_empty=True, compressed=False))
    write_format(format_out, n_symbols=n_symbols, sequence_size=window_size)
    write_unaligned(values=bitmaps, fout=bitmaps_out, block_size=block_size_bitmaps, symbol_size=n_symbols)


def main():
    from sys import stderr
    p = ArgumentParser()
    p.set_defaults(func=None)
    sp = p.add_subparsers()
    p_trace = sp.add_parser('trace-encode')
    p_cencode_model = sp.add_parser('sparse-to-c')
    p_cencode_trace = sp.add_parser('trace-to-c')
    p_create_sparse = sp.add_parser('create-model-sparse')
    p_verify_sparse = sp.add_parser('verify-model-sparse')
    p_create_bitmap = sp.add_parser('create-bitmap')
    p_verify_bitmap = sp.add_parser('verify-bitmap')
    p_create_bitmap.add_argument('-w', '--window-size', type=int_min(2), required=True)
    p_create_bitmap.add_argument('-B', '--bitmaps-out', type=FileType('wb'), required=True)
    p_create_bitmap.add_argument('sequences', type=FileType('rb'), nargs='+')
    p_create_bitmap.add_argument('-bst', '--block-size-trace', type=int, required=True, choices=[1, 2, 4, 8])
    p_create_bitmap.add_argument('-bsb', '--block-size-bitmaps', type=int, required=True, choices=[1, 2, 4, 8])
    p_create_bitmap.add_argument('-n', '--n-symbols', type=int_min(2), required=True)
    p_create_bitmap.add_argument('-F', '--format-out', type=FileType('wb'), required=True)
    p_create_bitmap.set_defaults(func=lambda args: create_bitmap(sequences_files=args.sequences,
                                                          n_symbols=args.n_symbols,
                                                          block_size_trace=args.block_size_trace,
                                                          block_size_bitmaps=args.block_size_bitmaps,
                                                          bitmaps_out=args.bitmaps_out,
                                                          window_size=args.window_size,
                                                          format_out=args.format_out
                                                          ))
    p_verify_bitmap.add_argument('sequences', type=FileType('rb'), nargs='+')
    p_verify_bitmap.add_argument('-w', '--window-size', type=int_min(2), required=True)
    p_verify_bitmap.add_argument('-bst', '--block-size-trace', type=int, required=True, choices=[1, 2, 4, 8])
    p_verify_bitmap.add_argument('-bsb', '--block-size-bitmaps', type=int, required=True, choices=[1, 2, 4, 8])
    p_verify_bitmap.add_argument('-f', '--format', type=FileType('rb'), required=True)
    p_verify_bitmap.add_argument('-b', '--bitmaps', type=FileType('rb'), required=True)
    p_verify_bitmap.set_defaults(func=lambda args: verify_bitmap(sequences_files=args.sequences,
                                                                 bitmaps_in=args.bitmaps,
                                                                 block_size_bitmaps=args.block_size_bitmaps,
                                                                 block_size_trace=args.block_size_trace,
                                                                 format_in=args.format,
                                                                 window_size=args.window_size))

    p_trace.add_argument('traces', type=FileType('rt'), nargs='+')
    p_trace.add_argument('-O', '--trace-out', type=FileType('wb'), required=False, default=None)
    p_trace.add_argument('-bs', '--block-size', type=int, required=True, choices=[1, 2, 4, 8])
    group = p_trace.add_mutually_exclusive_group(required=True)
    group.add_argument('-E', '--encoding-out', type=FileType('wb'), required=False, default=None)
    group.add_argument('-e', '--encoding-in', type=FileType('rb'), required=False, default=None)


    p_create_sparse.add_argument('-w', '--window-size', type=int_min(2), required=True)
    p_create_sparse.add_argument('-n', '--n-symbols', type=int_min(2), required=True)
    p_create_sparse.add_argument('-V', '--values-out', type=FileType('wb'), required=True)
    p_create_sparse.add_argument('-C', '--counters-out', type=FileType('wb'), required=True)
    p_create_sparse.add_argument('-bs', '--block-size', type=int, required=True, choices=[1, 2, 4, 8])
    p_create_sparse.add_argument('-F', '--format-out', type=FileType('wb'), required=True)
    p_create_sparse.add_argument('-S', '--skips-out', type=FileType('wb'), required=True)
    p_create_sparse.add_argument('-rs', '--rle-size', type=int_min(2), required=False, default=None)
    group = p_create_sparse.add_mutually_exclusive_group(required=True)
    group.add_argument('-ss', '--skips-size', type=int_min(0), default=None)
    group.add_argument('-sf', '--skips-root-factor', type=float_range(0, 1, exclude=True), default=None)

    p_create_sparse.add_argument('sequences', type=FileType('rb'), nargs='+')

    p_create_sparse.set_defaults(func=lambda args: create_sparse_model(sequences_files=args.sequences, window_size=args.window_size,
                                                         values_out=args.values_out, counters_out=args.counters_out,
                                                         block_size=args.block_size,
                                                         format_out=args.format_out,
                                                         n_symbols=args.n_symbols,
                                                         skips_size=args.skips_size,
                                                         skips_root_factor=args.skips_root_factor,
                                                         skips_out=args.skips_out,
                                                         rle_size=args.rle_size))

    # for p_verify_sparse in [p_verify_sparse]:
    p_verify_sparse.add_argument('sequences', type=FileType('rb'), nargs='+')
    p_verify_sparse.add_argument('-f', '--format', type=FileType('rb'), required=True)
    p_verify_sparse.add_argument('-v', '--values', type=FileType('rb'), required=True)
    p_verify_sparse.add_argument('-c', '--counters', type=FileType('rb'), required=True)
    p_verify_sparse.add_argument('-s', '--skips', type=FileType('rb'), required=True)
    p_verify_sparse.add_argument('-bs', '--block-size', type=int, required=True, choices=[1, 2, 4, 8])
    p_verify_sparse.add_argument('-rs', '--rle-size', type=int_min(2), required=False, default=None)
    # pp.add_argument('-n', '--n-symbols', type=int_min(2), required=True)
    p_verify_sparse.set_defaults(func=lambda args: verify_sparse_model(sequences_files=args.sequences,
                                                                       format_in=args.format,
                                                                       values_in=args.values,
                                                                       skips_in=args.skips,
                                                                       counters_in=args.counters,
                                                                       block_size=args.block_size,
                                                                       rle_size=args.rle_size))
    p_trace.set_defaults(func=lambda args: encode_trace(traces_files=args.traces,
                                                        block_size=args.block_size,
                                                        trace_out=args.trace_out,
                                                        encoding_in=args.encoding_in,
                                                        encoding_out=args.encoding_out))
    p_cencode_model.add_argument('-f', '--format', type=FileType('rb'), required=True)
    p_cencode_model.add_argument('-v', '--values', type=FileType('rb'), required=True)
    p_cencode_model.add_argument('-c', '--counters', type=FileType('rb'), required=True)
    p_cencode_model.add_argument('-s', '--skips', type=FileType('rb'), required=True)
    p_cencode_model.add_argument('-bs', '--block-size', type=int, required=True, choices=[1, 2, 4, 8])
    p_cencode_model.add_argument('-out', metavar='<data-structures-output>', type=FileType('w'), required=True)
    p_cencode_model.set_defaults(func=lambda args: model_to_c(
                                                        format_in=args.format,
                                                        values_in=args.values,
                                                        skips_in=args.skips,
                                                        counters_in=args.counters,
                                                        block_size=args.block_size,
                                                        data_out=args.out
                                                        ))
    p_cencode_trace.add_argument('-T', '--trace-out', metavar='<trace-output>', type=FileType('w'), required=True)
    p_cencode_trace.add_argument('-t', '--trace-in', type=FileType('rb'), required=True)
    p_cencode_trace.add_argument('-l', '--limit', type=int, required=False, default=None)
    p_cencode_trace.add_argument('-bs', '--block-size', type=int, required=True, choices=[1, 2, 4, 8])
    p_cencode_trace.set_defaults(func=lambda args: trace_to_c(
                                                        block_size=args.block_size,
                                                        trace_out=args.trace_out,
                                                        trace_in=args.trace_in,
                                                        limit=args.limit
                                                        ))
    args = p.parse_args()
    if not args.func:
        p.print_help(file=stderr)
        return 1
    return args.func(args)


if __name__ == '__main__':
    main()
