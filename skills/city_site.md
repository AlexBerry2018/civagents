# Skill city_site — where to found a city (Основать город)

Used by: economist, single. Input: settlers, their surroundings, allowed actions.
How it works: each tile around a settler gets points: Grassland or Plains 3, Hills 2, Forest 1, Desert/Tundra/Swamp/Mountains 0; bonuses River +2, Wheat +2, Pheasant/Wine/Silk/Resources +1; a tile with foreign units gets -5.
Decision rule: if "build city" is allowed and the current tile scores 3 or more, found the city now - waiting for a perfect site is a common mistake. Otherwise, if a neighbouring tile scores clearly higher, move there first. If "build city" is not allowed here, the settler is too close to an existing city: move 2-3 tiles away and try again.
The skill hint appears in the prompt as "[city_site] ...". Follow it unless you have a strong reason not to.
