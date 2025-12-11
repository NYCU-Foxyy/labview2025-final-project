#include <bits/stdc++.h>
using namespace std;

static const double M_PI = acos(-1);
static const int MAX_TEAMS = 16;
static const vector<pair<int, int>> steps{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

size_t height;
size_t width;
size_t mountain_density;
size_t city_density;

size_t team_count;
vector<int> team_sizes;

int game_zone_id;
vector<vector<char>> field;
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
vector<vector<pair<int, int>>> team_coords;
vector<vector<int>> id_field;

enum { SYMBOL_MOUNTAIN = '#', SYMBOL_CITY = '+', SYMBOL_EMPTY = '.', SYMBOL_GENERAL = 'A' };

struct DSU {
	int n;
	vector<int> p;
	vector<int> sz;
	
	void init(int _n) {
		n = _n;
		p.resize(n);
		sz.resize(n, 1);
		iota(p.begin(), p.end(), 0);
	}

	int find(int x) {
		return x == p[x] ? x : (p[x] = find(p[x]));
	}

	void unite(int x, int y) {
		int px = find(x);
		int py = find(y);
		if (px != py) {
			if (sz[px] < sz[py])
				swap(px, py);
			p[py] = px;
			sz[px] += sz[py];
		}
	}
};

int fill_id_field(int id, pair<int, int> src);
void generate();
void display();
void place_general_team(pair<int, int> origin, int team_id);
bool place_general(pair<int, int> location, int team_id);
bool coord_is_valid(int r, int c);

int fill_id_field(int id, pair<int, int> src) {
	queue<pair<int, int>> q;
	q.push(src);
	int cnt = 0;
	while (not q.empty()) {
		auto [r, c] = q.front();
		q.pop();
		if (id_field[r][c] != -1)
			continue;
		id_field[r][c] = id;
		cnt++;
		for (auto [dr, dc] : steps) {
			int nr = r + dr;
			int nc = c + dc;
			if (coord_is_valid(nr, nc) and field[nr][nc] == SYMBOL_EMPTY and id_field[nr][nc] == -1) {
				q.push(make_pair(nr, nc));
			}
		}
	}
	return cnt;
}

void generate() {
	field.clear();
	field.resize(height, vector<char>(width, SYMBOL_EMPTY));
	
	int mountain_count = 0.2 * mountain_density * 0.01 * height * width;
	int city_count = 0.08 * city_density * 0.01 * height * width;

	vector<pair<int, int>> mountain_coords;
	vector<pair<int, int>> coords;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			coords.push_back(make_pair(i, j));
		}
	}

	shuffle(coords.begin(), coords.end(), rng);

	for (int i = 0; i < mountain_count + city_count; i++) {
		auto [r, c] = coords.back();
		coords.pop_back();
		mountain_coords.push_back(make_pair(r, c));
		field[r][c] = SYMBOL_MOUNTAIN;
	}
	
	id_field.resize(height, vector<int>(width, -1));
	int id = 0;
	set<pair<int, int>> size_set;
	vector<int> sizes;
	for (int r = 0; r < height; r++) for (int c = 0; c < width; c++) {
		if (field[r][c] == SYMBOL_EMPTY and id_field[r][c] == -1) {
			int cnt = fill_id_field(id, make_pair(r, c));
			size_set.insert({cnt, id});
			sizes.push_back(cnt);
			id++;
		}
	}

	DSU dsu;
	dsu.init(size_set.size());

/*
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (id_field[i][j] == -1) {
				cout << "-";
			} else {
				int p = dsu.find(id_field[i][j]);
				if (p <= 9) {
					cout << p;
				} else {
					cout << (char)(p - 10 + 'A');
				}
			}
		}
		cout << endl;
	}
	for (auto [a, b] : size_set) {
		cout << a << ' ' << b << '\n';
	}
	getchar();
*/

	shuffle(mountain_coords.begin(), mountain_coords.end(), rng);

	bool connected_enough = false;

	for (int city_i = 0; city_i < city_count or not connected_enough; city_i++) {
		if (size_set.size() > 1 and not mountain_coords.empty()) {
			int zone_id = dsu.find(size_set.rbegin()->second);
			
			int adjacent_candidate = -1;
			int tunnel_candidate = -1;
			
			for (auto it = size_set.rbegin(); it != size_set.rend(); it++) {
				auto [size, candidate_zone_id] = *it;
				candidate_zone_id = dsu.find(candidate_zone_id);

				for (int i = 0; i < (int)mountain_coords.size(); i++) {
					int diff_id = -1;
					auto [r, c] = mountain_coords[i];
					
					bool is_adjacent = false;
					for (auto [dr, dc] : steps) {
						int nr = r + dr;
						int nc = c + dc;
						if (coord_is_valid(nr, nc) and field[nr][nc] != SYMBOL_MOUNTAIN) {
//							cerr << nr << ' ' << nc << ' ' << id_field[nr][nc] << '\n';
							if (int cell_id = dsu.find(id_field[nr][nc]); cell_id != candidate_zone_id) {
								diff_id = cell_id;
							} else {
								is_adjacent = true;
							}
						}
					} 
					if (is_adjacent and diff_id != -1) {
						tunnel_candidate = i;
//						cerr << r << ' ' << c << ", diff_id = " << diff_id << '\n';
						break;
					} else if (is_adjacent) {
						adjacent_candidate = i;
					}
				}
//				cerr << candidate_zone_id << ": " << tunnel_candidate << '\n';
				if (tunnel_candidate != -1)
					break;
			}
			
			int candidate = tunnel_candidate == -1 ? adjacent_candidate : tunnel_candidate;
			assert(candidate != -1);
			auto [r, c] = mountain_coords[candidate];

			if (city_i >= city_count)
				field[r][c] = SYMBOL_EMPTY;
			else
				field[r][c] = SYMBOL_CITY;
			
			if (tunnel_candidate != -1) {
				int p_zone_id = dsu.find(zone_id);
				
				int united_size = 1 + sizes[p_zone_id];

				size_set.erase(make_pair(sizes[p_zone_id], p_zone_id));
				
				for (auto [dr, dc] : steps) {
					int nr = r + dr;
					int nc = c + dc;
					if (not coord_is_valid(nr, nc))
						continue;
					if (field[nr][nc] == SYMBOL_MOUNTAIN)
						continue;
					int p_diff_id = dsu.find(id_field[nr][nc]);
					if (p_diff_id == p_zone_id)
						continue;
					size_set.erase(make_pair(sizes[p_diff_id], p_diff_id));
					dsu.unite(zone_id, p_diff_id);
					united_size += sizes[p_diff_id];
				}

				p_zone_id = dsu.find(zone_id);

				id_field[r][c] = p_zone_id;

				sizes[p_zone_id] = united_size;
				size_set.insert(make_pair(sizes[p_zone_id], p_zone_id));
			} else {
				int p_zone_id = dsu.find(zone_id);
				size_set.erase(make_pair(sizes[p_zone_id], p_zone_id));
				sizes[p_zone_id]++;
				size_set.insert(make_pair(sizes[p_zone_id], p_zone_id));

				id_field[r][c] = p_zone_id;
			}

			swap(mountain_coords[candidate], mountain_coords.back());
			mountain_coords.pop_back();
		} else {
			int r, c;
			bool valid;
			do {
				r = uniform_int_distribution<int>(0, height - 1)(rng);
				c = uniform_int_distribution<int>(0, width - 1)(rng);
				
				valid = false;
				for (auto [dr, dc] : steps) {
					int nr = r + dr;
					int nc = c + dc;
					if (coord_is_valid(nr, nc)) {
						if (field[nr][nc] != SYMBOL_MOUNTAIN) {
							valid = true;
							break;
						}
					}
				}

			} while(not valid);
			if (city_i >= city_count)
				field[r][c] = SYMBOL_EMPTY;
			else
				field[r][c] = SYMBOL_CITY;
		}

/*
		display();
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (id_field[i][j] == -1) {
					cerr << "-";
				} else {
					int p = dsu.find(id_field[i][j]);
					if (p <= 9) {
						cerr << p;
					} else {
						cerr << (char)(p - 10 + 'A');
					}
				}
			}
			cerr << endl;
		}
		for (auto [a, b] : size_set) {
			cerr << a << ' ' << b << '\n';
		}
		getchar();
*/

		if (size_set.size() >= 2 and (++size_set.rbegin())->first > 4) {
			connected_enough = false;
		} else {
			connected_enough = true;
		}
	}

	game_zone_id = size_set.rbegin()->second;
	for (int r = 0; r < height; r++) for (int c = 0; c < width; c++) if (id_field[r][c] != -1) {
		id_field[r][c] = dsu.find(id_field[r][c]);
	}

//	display();

	double center_row = height / 2;
	double center_column = width / 2;
	double scale_h = (double)height / 16 * 7;
	double scale_w = (double)width / 16 * 7;
	double offset = uniform_real_distribution<double>(0, 2 * M_PI)(rng);
	for (int team_i = 0; team_i < team_count; team_i++) {
		pair<int, int> origin;
		origin.first = sin(team_i * M_PI * 2 / team_count + offset) * scale_h + center_row;
		origin.second = cos(team_i * M_PI * 2 / team_count + offset) * scale_w + center_column;
		place_general_team(origin, team_i);
	}
}

void place_general_team(pair<int, int> origin, int team_id) {
	auto [sr, sc] = origin;
	int success_cnt = 0;
	int team_size = team_sizes[team_id];
//	cerr << "team #" << team_id << ": " << team_size << '\n';
	for (int player_i = 0; player_i < team_size; player_i++) {
		bool success = false;
		for (int radius = (player_i == 0 ? 0 : 2); radius < max(height / 2, width / 2); radius++) {
			vector<int> ops{0, 1, 2, 3};
			shuffle(ops.begin(), ops.end(), rng);
			for (auto op : ops) {
				if (op == 0) for (int d = 0; not success and d < radius; d++) {
					int r = sr + d;
					int c = sc + radius - d;
					if (place_general(make_pair(r, c), team_id)) {
						sr = r;
						sc = c;
						success = true;
						break;
					}
				}
				if (op == 1) for (int d = 0; not success and d < radius; d++) {
					int r = sr + radius - d;
					int c = sc - d;
					if (place_general(make_pair(r, c), team_id)) {
						sr = r;
						sc = c;
						success = true;
						break;
					}
				}
				if (op == 2) for (int d = 0; not success and d < radius; d++) {
					int r = sr - d;
					int c = sc - radius + d;
					if (place_general(make_pair(r, c), team_id)) {
						sr = r;
						sc = c;
						success = true;
						break;
					}
				}
				if (op == 3) for (int d = 0; not success and d < radius; d++) {
					int r = sr - radius + d;
					int c = sc + d;
					if (place_general(make_pair(r, c), team_id)) {
						sr = r;
						sc = c;
						success = true;
						break;
					}
				}
			}
//			cerr << "radius = " << radius << '\n';
			if (success) {
				break;
			}
		}
		if (not success) {
//			display();
			cerr << "Failed to place general." << endl;
			exit(EXIT_FAILURE);
		}
	}
}

bool place_general(pair<int, int> location, int team_id) {
	auto [r, c] = location;

//	cerr << "trying (" << r << ", " << c << ")\n";
	
	if (not (coord_is_valid(r, c))) {
		return false;
	}
	if (id_field[r][c] != game_zone_id) {
		return false;
	}
	if (field[r][c] != SYMBOL_EMPTY) {
		return false;
	}

	for (auto [dr, dc] : steps) {
		int nr = r + dr;
		int nc = c + dc;
		if (not coord_is_valid(nr, nc)) continue;
		if (field[nr][nc] >= SYMBOL_GENERAL and field[nr][nc] < SYMBOL_GENERAL + MAX_TEAMS)
			return false;
	}
		
	vector<vector<bool>> vst(height, vector<bool>(width, false));
	auto get_zone_size = [&](int sr, int sc) {
		queue<pair<int, int>> q;
		if (field[sr][sc] != SYMBOL_MOUNTAIN and not (field[sr][sc] >= SYMBOL_GENERAL and field[sr][sc] < SYMBOL_GENERAL + MAX_TEAMS))
			q.push(make_pair(sr, sc));
		int cnt = 0;
		bool adj = false;
		bool adj_opp = false;
		if (field[sr][sc] == SYMBOL_GENERAL + team_id)
			adj = true;
		while (not q.empty()) {
			auto [r, c] = q.front();
			q.pop();
			if (vst[r][c]) continue;
			vst[r][c] = true;

			cnt++;

			for (auto [dr, dc] : steps) {
				int nr = r + dr;
				int nc = c + dc;
				if (not coord_is_valid(nr, nc))
					continue;
				if (field[nr][nc] == SYMBOL_MOUNTAIN)
					continue;
				if (field[nr][nc] >= SYMBOL_GENERAL and field[nr][nc] < SYMBOL_GENERAL + MAX_TEAMS) {
					if (not (nr == location.first and nc == location.second)) {
						if (field[nr][nc] == SYMBOL_GENERAL + team_id)
							adj = true;
						else
							adj_opp = true;
						continue;
					}
					continue;
				}
				q.push(make_pair(nr, nc));
			}
		}
		return make_tuple(cnt, adj, adj_opp);
	};

	auto mark_with_id = [&](vector<vector<int>>& local_id_field, int sr, int sc, int id) {
		queue<pair<int, int>> q;
		q.push(make_pair(sr, sc));
		int cnt = 0;
		while (not q.empty()) {
			auto [r, c] = q.front();
			q.pop();
			if (local_id_field[r][c] == id)
				continue;
			local_id_field[r][c] = id;
			cnt++;

			for (auto [dr, dc] : steps) {
				int nr = r + dr;
				int nc = c + dc;
				if (not coord_is_valid(nr, nc))
					continue;
				if (field[nr][nc] == SYMBOL_EMPTY)
					q.push(make_pair(nr, nc));
			}
		}
		return cnt;
	};

	field[r][c] = SYMBOL_GENERAL + team_id;
	vector<tuple<int, bool, pair<int, int>>> zone_sizes;
	bool adj_opp = false;
	for (auto [dr, dc] : steps) {
		int nr = r + dr;
		int nc = c + dc;
		if (not coord_is_valid(nr, nc)) {
			zone_sizes.push_back(make_tuple(0, false, make_pair(dr, dc)));
			continue;
		}
		auto [sz, adj, ret_adj_opp] = get_zone_size(nr, nc);
		adj_opp = adj_opp or ret_adj_opp;
		zone_sizes.push_back(make_tuple(sz, adj, make_pair(dr, dc)));
	}
	
	if (team_id != 0 and not adj_opp) {
		field[r][c] = SYMBOL_EMPTY;
		return false;
	}

	sort(zone_sizes.begin(), zone_sizes.end());	

/*
	cerr << "place at (" << r << ", " << c << "):\n";
	for (auto [sz, adj, step] : zone_sizes) {
		cerr << sz << ' ' << adj << " (" << step.first << ", " << step.second << ")\n";
	}
	cerr << '\n';
*/

	auto local_id_field = id_field;
	for (int r = 0; r < height; r++) for (int c = 0; c < width; c++) {
		local_id_field[r][c] = -2;
	}
	int marked_cnt = mark_with_id(local_id_field, r + get<2>(zone_sizes[3]).first, c + get<2>(zone_sizes[3]).second, game_zone_id);
	if (marked_cnt < height * width / 4) {
		field[r][c] = SYMBOL_EMPTY;
		return false;
	}

/*
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			auto x = local_id_field[i][j];
			if (x == -1) {
				cerr << "-";
			} else if (x == -2) {
				cerr << "=";
			} else {
				if (x <= 9) {
					cerr << x;
				} else {
					cerr << (char)(x - 10 + 'A');
				}
			}
		}
		cerr << endl;
	}
*/

//	display();

	for (auto [sz, adj, step] : zone_sizes) {
		int nr = r + step.first;
		int nc = c + step.second;
		if (not coord_is_valid(nr, nc)) continue;
		if (local_id_field[nr][nc] == -2 and (sz > 4 or adj)) { 
			field[r][c] = SYMBOL_EMPTY;
			return false;
		}
	}

//	cerr << "SUCCESS\n";
	return true;
}

void display() {
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			cout << field[r][c];
		}
		cout << endl;
	}
}

bool coord_is_valid(int r, int c) {
	return 0 <= r and r < height and 0 <= c and c < width;
}

int main(int argc, char* argv[]) {

	cin >> height >> width >> mountain_density >> city_density >> team_count;
/*
	height = atoi(argv[1]);
	width = atoi(argv[2]);
	mountain_density = atoi(argv[3]);
	city_density = atoi(argv[4]);
	team_count = atoi(argv[5]);
*/

	for (int i = 0; i < team_count; i++) {
//		team_sizes.push_back(atoi(argv[6 + i]));
		int team_size;
		cin >> team_size;
		team_sizes.push_back(team_size);
	}

	generate();
	display();

	return 0;
}
