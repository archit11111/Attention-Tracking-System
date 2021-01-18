import React, { useState, useEffect, useContext } from "react";

import { Container } from "react-bootstrap";
import { useHistory } from "react-router-dom";
import UserContext from "../../context/UserContext";
import Axios from "axios";
import ErrorNotice from "../misc/ErrorNotice";

import "./auth.css";

export default function Register() {
	const [email, setEmail] = useState();
	const [password, setPassword] = useState();
	const [passwordCheck, setPasswordCheck] = useState();
	const [displayName, setDisplayName] = useState();
	const [error, setError] = useState();
	const [isAdmin, setIsAdmin] = useState(false);
	const [adminKey, setAdminKey] = useState();

	const { userData, setUserData } = useContext(UserContext);
	const history = useHistory();

	useEffect(() => {
		if (userData.user) history.push("/");
	}, [history, userData.user]);

	const submit = async (e) => {
		e.preventDefault();
		let serverUrl = "http://localhost:10000/users/register";
		if (isAdmin) serverUrl = "http://localhost:10000/users/register/admin";
		try {
			const newUser = { email, password, passwordCheck, displayName, adminKey };
			console.log("--->", email, password);
			await Axios.post(serverUrl, newUser);
			const loginRes = await Axios.post(`http://localhost:10000/users/login/${isAdmin ? "admin" : ""}`, {
				email,
				password,
			});
			setUserData({
				token: loginRes.data.token,
				user: loginRes.data.user,
			});
			localStorage.setItem("auth-token", loginRes.data.token);
			history.push("/");
		} catch (err) {
			err.response.data.msg && setError(err.response.data.msg);
		}
	};

	return (
		<Container>
			<h2>Register</h2>
			{error && <ErrorNotice message={error} clearError={() => setError(undefined)} />}
			<form className="form" onSubmit={submit}>
				<label htmlFor="register-email">Email</label>
				<input id="register-email" type="email" onChange={(e) => setEmail(e.target.value)} />

				<label htmlFor="register-password">Password</label>
				<input id="register-password" type="password" onChange={(e) => setPassword(e.target.value)} />
				<input
					type="password"
					placeholder="Verify password"
					onChange={(e) => setPasswordCheck(e.target.value)}
				/>

				<label htmlFor="register-display-name">Display name</label>
				<input id="register-display-name" type="text" onChange={(e) => setDisplayName(e.target.value)} />

				<label htmlFor="admin"> Register as Admin</label>
				<input
					type="checkbox"
					id="admin"
					name="admin"
					value="admin"
					onChange={(e) => {
						setIsAdmin(e.target.checked);
					}}
				/>

				{isAdmin && (
					<>
						<label htmlFor="admin-key">Admin Key</label>
						<input id="admin-key" type="password" onChange={(e) => setAdminKey(e.target.value)} />
					</>
				)}

				<input type="submit" value="Register" />
			</form>
		</Container>
	);
}
